"""Question validation before JSON encoding."""

from collections.abc import Mapping, Sequence

from typesafe_sdk._core.errors import TypeSafeError
from typesafe_sdk._core.json_types import JSONContent
from typesafe_sdk._core.question_types import Choice, Noul, Question, Score


def normalize_questions(questions: Mapping[str, Question]) -> dict[str, Question]:
    if not questions:
        raise TypeSafeError("At least one question is required.")
    for name, question in questions.items():
        if isinstance(question, Score):
            _validate_score_criteria(name, question.criteria)
        elif isinstance(question, Choice):
            _validate_choice_criteria(name, question.criteria)
        elif not isinstance(question, Noul):
            if not isinstance(question, dict) or not isinstance(question.get("type"), str) or not question["type"]:
                raise TypeSafeError(f'Question "{name}" must be a question object or a dictionary with a nonempty string "type".')
            if question["type"] in ("choice", "score") and "criteria" not in question:
                raise TypeSafeError(f'Question "{name}" requires "criteria".')
            if question["type"] == "score":
                _validate_score_criteria(name, question["criteria"])
            elif question["type"] == "choice":
                _validate_choice_criteria(name, question["criteria"])
    return dict(questions)


def _validate_score_criteria(name: str, criteria: Sequence[JSONContent]) -> None:
    """Reject a score question with no criteria; at least one score is required."""
    if not criteria:
        raise TypeSafeError(f'Score question "{name}" has no criteria; at least one score is required.')


def _validate_choice_criteria(name: str, criteria: Mapping[str, JSONContent | None]) -> None:
    """Reject a choice question with no criteria; at least one choice is required."""
    if not criteria:
        raise TypeSafeError(f'Choice question "{name}" has no criteria; at least one choice is required.')
