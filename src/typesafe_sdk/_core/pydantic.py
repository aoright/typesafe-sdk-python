"""Conversion between flat Pydantic models and System One questions."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal, TypeVar, get_args, get_origin

from pydantic import BaseModel

from typesafe_sdk._core.errors import TypeSafeError
from typesafe_sdk._core.json_types import JSONContent
from typesafe_sdk._core.question_types import (
    Choice,
    Noul,
    NoulCriteria,
    Question,
    Score,
)
from typesafe_sdk._core.questions import normalize_questions
from typesafe_sdk._core.response_types import (
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    SystemOneResponse,
)
from typesafe_sdk._core.schemas.base import _ResponseMixin

ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True)
class QuestionConfig:
    """Configuration for question extraction from a Pydantic model field."""

    instructions: JSONContent | None = None
    criteria: Mapping[str, JSONContent | None] | Sequence[str] | NoulCriteria | None = None


@dataclass(frozen=True)
class ChoiceConfig(QuestionConfig):
    """Configuration for a categorical choice question."""


@dataclass(frozen=True)
class NoulConfig(QuestionConfig):
    """Configuration for a yes/no boolean question."""


@dataclass(frozen=True)
class ScoreConfig:
    """Configuration for a score question with an ordered rubric."""

    instructions: JSONContent | None = None
    levels: Sequence[JSONContent] | None = None
    criteria: Sequence[JSONContent] | None = None

    def __post_init__(self) -> None:
        if self.criteria is not None and self.levels is not None and list(self.criteria) != list(self.levels):
            raise TypeSafeError("Cannot specify different values for both 'levels' and 'criteria'.")
        if self.criteria is None and self.levels is not None:
            object.__setattr__(self, "criteria", self.levels)
        elif self.levels is None and self.criteria is not None:
            object.__setattr__(self, "levels", self.criteria)


def _normalize_choice_criteria(
    criteria: Mapping[str, JSONContent | None] | Sequence[str] | None,
) -> dict[str, JSONContent | None]:
    if criteria is None:
        return {}
    if isinstance(criteria, Mapping):
        return dict(criteria)
    if isinstance(criteria, Sequence) and not isinstance(criteria, (str, bytes)):
        return dict.fromkeys(criteria)
    raise TypeSafeError("Choice criteria must be a mapping or a sequence of labels.")


def resolve_system_one_input(state: JSONContent | BaseModel | None, input: JSONContent | BaseModel | None) -> JSONContent:
    """Resolve the input alias and serialize model instances as JSON values."""
    if state is not None and input is not None:
        raise TypeSafeError("Pass either input or state, not both.")
    value = input if input is not None else state
    if value is None:
        raise TypeSafeError("An input or state is required.")
    if isinstance(value, BaseModel):
        serialized = value.model_dump(mode="json")
        if not isinstance(serialized, (str, dict, list)):
            raise TypeSafeError("Input models must serialize to text, an object, or an array.")
        return serialized
    return value


def questions_from_model(model: type[BaseModel]) -> dict[str, Question]:
    """Convert supported model fields to questions, using Python field names."""
    if not isinstance(model, type) or not issubclass(model, BaseModel):
        raise TypeSafeError("response_model must be a Pydantic BaseModel subclass.")
    if issubclass(model, SystemOneResponse):
        raise TypeSafeError(
            "When questions is omitted, response_model cannot be SystemOneResponse; provide a flat Pydantic model describing the desired answers."
        )
    if getattr(model, "__pydantic_root_model__", False) or not model.model_fields:
        raise TypeSafeError("response_model must be a nonempty flat model, not a RootModel.")

    questions: dict[str, Question] = {}
    for name, field in model.model_fields.items():
        annotation = field.annotation
        metadata = [
            item
            for item in field.metadata
            if isinstance(item, (QuestionConfig, ScoreConfig, Noul, Choice, Score))
        ]
        if len(metadata) > 1:
            raise TypeSafeError(f'Field "{name}" has conflicting question metadata.')
        config = metadata[0] if metadata else None

        is_choice_answer = isinstance(annotation, type) and issubclass(annotation, ChoiceAnswer)
        is_noul_answer = isinstance(annotation, type) and issubclass(annotation, NoulAnswer)
        is_score_answer = isinstance(annotation, type) and issubclass(annotation, ScoreAnswer)
        is_int_type = annotation is int or (
            isinstance(annotation, type) and issubclass(annotation, int) and not issubclass(annotation, bool)
        )

        labels: tuple[str, ...] | None = None
        if get_origin(annotation) is Literal:
            raw_labels = get_args(annotation)
            if not raw_labels or not all(isinstance(label, str) for label in raw_labels):
                raise TypeSafeError(f'Field "{name}" requires nonempty string Literal or Enum values.')
            labels = tuple(raw_labels)
        elif isinstance(annotation, type) and issubclass(annotation, Enum):
            raw_labels = tuple(member.value for member in annotation)
            if not raw_labels or not all(isinstance(label, str) for label in raw_labels):
                raise TypeSafeError(f'Field "{name}" requires nonempty string Literal or Enum values.')
            labels = raw_labels

        question: Question
        if labels is not None:
            if config is None:
                question = Choice(criteria=dict.fromkeys(labels), instructions=field.description)
            elif isinstance(config, Choice):
                if set(config.criteria) != set(labels):
                    raise TypeSafeError(f'Field "{name}" has incompatible Choice metadata.')
                question = config
                if question.instructions is None and field.description is not None:
                    question = question.model_copy(update={"instructions": field.description})
            elif isinstance(config, (QuestionConfig, ChoiceConfig)):
                if config.criteria is not None:
                    norm_criteria = _normalize_choice_criteria(config.criteria)
                    if set(norm_criteria.keys()) != set(labels):
                        raise TypeSafeError(f'Field "{name}" has incompatible Choice metadata.')
                    criteria = norm_criteria
                else:
                    criteria = dict.fromkeys(labels)
                instructions = config.instructions if config.instructions is not None else field.description
                question = Choice(criteria=criteria, instructions=instructions)
            else:
                raise TypeSafeError(f'Field "{name}" has incompatible Choice metadata.')

        elif is_choice_answer or (
            annotation is str
            and isinstance(config, (Choice, ChoiceConfig, QuestionConfig))
            and getattr(config, "criteria", None) is not None
        ):
            if isinstance(config, Choice):
                question = config
                if question.instructions is None and field.description is not None:
                    question = question.model_copy(update={"instructions": field.description})
            elif isinstance(config, (QuestionConfig, ChoiceConfig)) and config.criteria is not None:
                instructions = config.instructions if config.instructions is not None else field.description
                criteria = _normalize_choice_criteria(config.criteria)
                question = Choice(criteria=criteria, instructions=instructions)
            else:
                raise TypeSafeError(f'Field "{name}" requires criteria in Choice or QuestionConfig.')

        elif annotation is bool or is_noul_answer or (annotation is float and not isinstance(config, (ScoreConfig, Score))):
            if config is None:
                question = Noul(instructions=field.description)
            elif isinstance(config, Noul):
                question = config
                if question.instructions is None and field.description is not None:
                    question = question.model_copy(update={"instructions": field.description})
            elif isinstance(config, (QuestionConfig, NoulConfig)):
                instructions = config.instructions if config.instructions is not None else field.description
                criteria = config.criteria if isinstance(config.criteria, dict) else None
                question = Noul(instructions=instructions, criteria=criteria)
            else:
                raise TypeSafeError(f'Field "{name}" requires Noul metadata.')

        elif is_score_answer or ((is_int_type or annotation is float) and isinstance(config, (ScoreConfig, Score))):
            if config is None:
                raise TypeSafeError(f'Field "{name}" requires ScoreConfig or Score metadata.')
            if isinstance(config, Score):
                if not config.criteria:
                    raise TypeSafeError(f'Score question "{name}" has no criteria.')
                question = config
                if question.instructions is None and field.description is not None:
                    question = question.model_copy(update={"instructions": field.description})
            elif isinstance(config, ScoreConfig):
                criteria_list = config.criteria or config.levels
                if not criteria_list:
                    raise TypeSafeError(f'Score question "{name}" has no criteria.')
                instructions = config.instructions if config.instructions is not None else field.description
                question = Score(instructions=instructions, criteria=criteria_list)
            else:
                raise TypeSafeError(f'Field "{name}" requires Score metadata.')

        else:
            raise TypeSafeError(
                f'Field "{name}" has an unsupported type; use string Literal/Enum, bool, or float with Noul/Score metadata.'
            )

        questions[name] = question

    return normalize_questions(questions)


def parse_model(model: type[ModelT], response: SystemOneResponse) -> ModelT:
    """Require every requested answer, then run Pydantic model validation."""
    values: dict[str, object] = {}
    for name, question in questions_from_model(model).items():
        answer = response.answers.get(name)
        if answer is None:
            raise TypeSafeError(f'Missing answer for field "{name}".')
        field = model.model_fields[name]
        annotation = field.annotation

        is_choice_answer = isinstance(annotation, type) and issubclass(annotation, ChoiceAnswer)
        is_noul_answer = isinstance(annotation, type) and issubclass(annotation, NoulAnswer)
        is_score_answer = isinstance(annotation, type) and issubclass(annotation, ScoreAnswer)
        is_int_type = annotation is int or (
            isinstance(annotation, type) and issubclass(annotation, int) and not issubclass(annotation, bool)
        )

        if isinstance(question, Choice):
            if not isinstance(answer, ChoiceAnswer):
                raise TypeSafeError(f'Wrong answer type for field "{name}".')
            if answer.choice not in question.criteria:
                raise TypeSafeError(f'Unknown choice for field "{name}": {answer.choice!r}.')
            if is_choice_answer:
                if annotation is ChoiceAnswer:
                    values[name] = answer
                else:
                    values[name] = annotation.model_validate(answer.model_dump())
            elif isinstance(annotation, type) and issubclass(annotation, Enum):
                values[name] = annotation(answer.choice)
            else:
                values[name] = answer.choice

        elif isinstance(question, Noul):
            if not isinstance(answer, NoulAnswer):
                raise TypeSafeError(f'Wrong answer type for field "{name}".')
            if math.isnan(answer.noul) or math.isinf(answer.noul) or not (-1e-6 <= answer.noul <= 1.0 + 1e-6):
                raise TypeSafeError(f'Invalid probability for field "{name}"; expected a value from zero to one.')
            if is_noul_answer:
                if annotation is NoulAnswer:
                    values[name] = answer
                else:
                    values[name] = annotation.model_validate(answer.model_dump())
            elif annotation is bool:
                values[name] = answer.noul >= 0.5
            else:
                values[name] = answer.noul

        elif isinstance(question, Score):
            if not isinstance(answer, ScoreAnswer):
                raise TypeSafeError(f'Wrong answer type for field "{name}".')
            if math.isnan(answer.score) or math.isinf(answer.score) or not (-1e-6 <= answer.score <= (len(question.criteria) - 1) + 1e-6):
                raise TypeSafeError(f'Invalid score for field "{name}"; expected a value within the rubric.')
            if is_score_answer:
                if annotation is ScoreAnswer:
                    values[name] = answer
                else:
                    values[name] = annotation.model_validate(answer.model_dump())
            elif is_int_type:
                values[name] = round(answer.score)
            else:
                values[name] = answer.score

        else:
            raise TypeSafeError(f'Wrong answer type for field "{name}".')

    result = model.model_validate(values, by_name=True, by_alias=False)
    if isinstance(result, _ResponseMixin):
        result.__dict__["_request_id"] = response.request_id
        result.__dict__["_raw"] = response.raw_http_response
    return result
