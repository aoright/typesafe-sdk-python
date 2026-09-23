"""Tests for simplified question definition and output handling via Pydantic models."""

import json
from enum import Enum
from typing import Annotated, Any, Literal

import httpx2
import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    ValidationError,
    field_validator,
)

from tests.conftest import ClientFactory
from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    ChoiceAnswer,
    ChoiceConfig,
    Noul,
    NoulAnswer,
    NoulConfig,
    QuestionConfig,
    Score,
    ScoreAnswer,
    ScoreConfig,
    SystemOneResponse,
    TypeSafeError,
    Usage,
)
from typesafe_sdk._core.pydantic import (
    parse_model,
    questions_from_model,
    resolve_system_one_input,
)


class TicketCategory(Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    OTHER = "other"


class Ticket(BaseModel):
    category: Annotated[
        Literal["billing", "technical", "other"],
        QuestionConfig(
            instructions={
                "task": "Classify the ticket",
                "rules": ["Use billing for charges, invoices, and refunds"],
            },
        ),
    ] = Field(description="The ticket's category")

    urgent: Annotated[
        bool,
        QuestionConfig(instructions="Does this need immediate attention?"),
    ]

    severity: Annotated[
        ScoreAnswer,
        ScoreConfig(
            instructions={
                "task": "Assess the severity",
                "consider": ["Impact on the customer", "Available workarounds"],
            },
            levels=(
                "Minor inconvenience",
                "Degraded service",
                "Completely blocked",
            ),
        ),
    ] = Field(description="Severity assessment with rubric and probabilities")


def make_system_one_response(answers: dict[str, Any]) -> SystemOneResponse:
    return SystemOneResponse(model="jev-latest", usage=Usage(input_tokens=10, output_tokens=5), answers=answers)


def test_issue_5_question_extraction() -> None:
    questions = questions_from_model(Ticket)
    assert len(questions) == 3

    category_q = questions["category"]
    assert isinstance(category_q, Choice)
    assert set(category_q.criteria.keys()) == {"billing", "technical", "other"}
    assert category_q.instructions == {
        "task": "Classify the ticket",
        "rules": ["Use billing for charges, invoices, and refunds"],
    }

    urgent_q = questions["urgent"]
    assert isinstance(urgent_q, Noul)
    assert urgent_q.instructions == "Does this need immediate attention?"

    severity_q = questions["severity"]
    assert isinstance(severity_q, Score)
    assert list(severity_q.criteria) == [
        "Minor inconvenience",
        "Degraded service",
        "Completely blocked",
    ]
    assert severity_q.instructions == {
        "task": "Assess the severity",
        "consider": ["Impact on the customer", "Available workarounds"],
    }


def test_issue_5_response_parsing() -> None:
    raw_response = make_system_one_response(
        {
            "category": ChoiceAnswer(choice="billing", confidence=0.95, probabilities={"billing": 0.95, "technical": 0.05, "other": 0.0}),
            "urgent": NoulAnswer(noul=0.88),
            "severity": ScoreAnswer(
                score=1.0,
                confidence=0.8,
                probabilities={0: 0.1, 1: 0.8, 2: 0.1},
                legend={0: "Minor inconvenience", 1: "Degraded service", 2: "Completely blocked"},
            ),
        }
    )
    ticket = parse_model(Ticket, raw_response)
    assert ticket.category == "billing"
    assert ticket.urgent is True
    assert isinstance(ticket.severity, ScoreAnswer)
    assert ticket.severity.score == 1.0


async def test_issue_5_client_round_trip(clients: ClientFactory) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        data = json.loads(request.content)
        assert data["state"] == "I was charged twice. Please fix this today."
        assert data["model"] == "jev-latest"
        assert set(data["questions"].keys()) == {"category", "urgent", "severity"}
        return httpx2.Response(
            200,
            json={
                "model": "jev-latest",
                "usage": {"input_tokens": 12, "output_tokens": 3},
                "answers": {
                    "category": {"type": "choice", "choice": "billing", "confidence": 0.9, "probabilities": {"billing": 0.9, "technical": 0.1, "other": 0.0}},
                    "urgent": {"type": "noul", "noul": 0.85},
                    "severity": {
                        "type": "score",
                        "score": 2.0,
                        "confidence": 0.9,
                        "legend": {"0": "Minor inconvenience", "1": "Degraded service", "2": "Completely blocked"},
                        "probabilities": {"0": 0.05, "1": 0.05, "2": 0.9},
                    },
                },
            },
        )

    client = clients(handler)
    if isinstance(client, AsyncTypeSafeClient):
        ticket = await client.system_one(
            state="I was charged twice. Please fix this today.",
            response_model=Ticket,
        )
    else:
        ticket = client.system_one(
            state="I was charged twice. Please fix this today.",
            response_model=Ticket,
        )

    assert isinstance(ticket, Ticket)
    assert ticket.category == "billing"
    assert ticket.urgent is True
    assert ticket.severity.score == 2.0


class TerserModel(BaseModel):
    is_urgent: bool = Field(description="Does this require immediate action?")
    sentiment: Literal["positive", "neutral", "negative"] = Field(description="Overall sentiment of the message")


def test_terse_field_description_fallback() -> None:
    questions = questions_from_model(TerserModel)
    assert questions["is_urgent"] == Noul(instructions="Does this require immediate action?")
    assert questions["sentiment"] == Choice(
        criteria={"positive": None, "neutral": None, "negative": None},
        instructions="Overall sentiment of the message",
    )


class EnumModel(BaseModel):
    category: TicketCategory = Field(description="Categorize ticket")


def test_enum_field_handling() -> None:
    questions = questions_from_model(EnumModel)
    assert questions["category"] == Choice(
        criteria={"billing": None, "technical": None, "other": None},
        instructions="Categorize ticket",
    )
    result = parse_model(
        EnumModel,
        make_system_one_response(
            {"category": ChoiceAnswer(choice="technical", confidence=1.0, probabilities={"technical": 1.0})}
        ),
    )
    assert result.category is TicketCategory.TECHNICAL


class TypesModel(BaseModel):
    flag: bool
    prob: Annotated[float, NoulConfig(instructions="P")]
    quality: Annotated[float, ScoreConfig(levels=["bad", "good"])]
    raw_choice: Annotated[ChoiceAnswer, ChoiceConfig(criteria={"yes": None, "no": None})]
    raw_noul: NoulAnswer


def test_various_output_types() -> None:
    questions = questions_from_model(TypesModel)
    assert isinstance(questions["flag"], Noul)
    assert isinstance(questions["prob"], Noul)
    assert isinstance(questions["quality"], Score)
    assert isinstance(questions["raw_choice"], Choice)
    assert isinstance(questions["raw_noul"], Noul)

    result = parse_model(
        TypesModel,
        make_system_one_response(
            {
                "flag": NoulAnswer(noul=0.4),
                "prob": NoulAnswer(noul=0.75),
                "quality": ScoreAnswer(score=1.0, confidence=1.0, probabilities={0: 0.0, 1: 1.0}, legend={0: "bad", 1: "good"}),
                "raw_choice": ChoiceAnswer(choice="yes", confidence=1.0, probabilities={"yes": 1.0}),
                "raw_noul": NoulAnswer(noul=0.9),
            }
        ),
    )
    assert result.flag is False
    assert result.prob == 0.75
    assert result.quality == 1.0
    assert result.raw_choice.choice == "yes"
    assert result.raw_noul.noul == 0.9


def test_score_config_levels_and_criteria() -> None:
    sc1 = ScoreConfig(levels=["a", "b"])
    assert sc1.criteria == ["a", "b"]
    assert sc1.levels == ["a", "b"]

    sc2 = ScoreConfig(criteria=["x", "y"])
    assert sc2.levels == ["x", "y"]

    with pytest.raises(TypeSafeError, match="different values"):
        ScoreConfig(levels=["a"], criteria=["b"])


def test_score_config_empty_criteria() -> None:
    class EmptyScore(BaseModel):
        score: Annotated[ScoreAnswer, ScoreConfig(levels=[])]

    with pytest.raises(TypeSafeError, match="no criteria"):
        questions_from_model(EmptyScore)


def test_conflicting_metadata() -> None:
    class Conflict(BaseModel):
        val: Annotated[bool, NoulConfig(), ScoreConfig(levels=["a", "b"])]

    with pytest.raises(TypeSafeError, match="conflicting"):
        questions_from_model(Conflict)


@pytest.mark.parametrize(
    "annotation",
    [str, int, list[str], dict[str, str], Literal[1, 2], RootModel[bool]],
)
def test_unsupported_types(annotation: Any) -> None:
    class BadModel(BaseModel):
        field: annotation

    with pytest.raises(TypeSafeError):
        questions_from_model(BadModel)


def test_incompatible_choice_criteria() -> None:
    class IncompatibleChoice(BaseModel):
        cat: Annotated[Literal["a", "b"], QuestionConfig(criteria={"c": None})]

    with pytest.raises(TypeSafeError, match="incompatible"):
        questions_from_model(IncompatibleChoice)


def test_missing_answer_raises() -> None:
    class Simple(BaseModel):
        ans: bool

    with pytest.raises(TypeSafeError, match="Missing answer"):
        parse_model(Simple, make_system_one_response({}))


def test_unknown_choice_raises() -> None:
    class SimpleChoice(BaseModel):
        val: Literal["a", "b"]

    with pytest.raises(TypeSafeError, match="Unknown choice"):
        parse_model(
            SimpleChoice,
            make_system_one_response({"val": ChoiceAnswer(choice="unknown", confidence=1.0, probabilities={})}),
        )


@pytest.mark.parametrize("prob", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_noul_probability(prob: float) -> None:
    class SimpleNoul(BaseModel):
        val: bool

    with pytest.raises(TypeSafeError, match="Invalid probability"):
        parse_model(SimpleNoul, make_system_one_response({"val": NoulAnswer(noul=prob)}))


@pytest.mark.parametrize("score", [-0.1, 2.1, float("nan"), float("inf")])
def test_invalid_score_range(score: float) -> None:
    class SimpleScore(BaseModel):
        val: Annotated[ScoreAnswer, ScoreConfig(levels=["l1", "l2"])]

    with pytest.raises(TypeSafeError, match="Invalid score"):
        parse_model(
            SimpleScore,
            make_system_one_response(
                {"val": ScoreAnswer(score=score, confidence=1.0, probabilities={}, legend={})}
            ),
        )


def test_field_validators_run() -> None:
    class Validated(BaseModel):
        choice: Literal["a", "b"]

        @field_validator("choice")
        @classmethod
        def reject_b(cls, val: str) -> str:
            if val == "b":
                raise ValueError("b rejected by custom validator")
            return val

    with pytest.raises(ValidationError, match="b rejected"):
        parse_model(
            Validated,
            make_system_one_response({"choice": ChoiceAnswer(choice="b", confidence=1.0, probabilities={})}),
        )


class StateInputModel(BaseModel):
    message: str
    code: int


def test_resolve_system_one_input() -> None:
    assert resolve_system_one_input("str", None) == "str"
    assert resolve_system_one_input(None, "str") == "str"
    assert resolve_system_one_input({"k": "v"}, None) == {"k": "v"}

    model = StateInputModel(message="hello", code=42)
    assert resolve_system_one_input(model, None) == {"message": "hello", "code": 42}
    assert resolve_system_one_input(None, model) == {"message": "hello", "code": 42}

    with pytest.raises(TypeSafeError, match="Pass either input or state, not both"):
        resolve_system_one_input("a", "b")

    with pytest.raises(TypeSafeError, match="An input or state is required"):
        resolve_system_one_input(None, None)

    with pytest.raises(TypeSafeError, match="Input models must serialize"):
        resolve_system_one_input(RootModel[int](1), None)


async def test_inferred_mode_safety_checks(clients: ClientFactory) -> None:
    client = clients(lambda _: httpx2.Response(200, json={}))

    # Extra body overriding state or questions in inferred mode must fail before network
    with pytest.raises(TypeSafeError, match="extra_body cannot override"):
        if isinstance(client, AsyncTypeSafeClient):
            await client.system_one(state="x", response_model=Ticket, extra_body={"state": "y"})
        else:
            client.system_one(state="x", response_model=Ticket, extra_body={"state": "y"})

    # SystemOneResponse without questions must fail
    with pytest.raises(TypeSafeError, match="cannot be SystemOneResponse"):
        if isinstance(client, AsyncTypeSafeClient):
            await client.system_one(state="x", response_model=SystemOneResponse)
        else:
            client.system_one(state="x", response_model=SystemOneResponse)

    # Neither questions nor response_model
    call = getattr(client, "system_one")  # noqa: B009
    with pytest.raises(TypeSafeError, match="Pass questions or a response_model"):
        if isinstance(client, AsyncTypeSafeClient):
            await call(state="x")
        else:
            call(state="x")


def test_float_without_metadata_maps_to_noul() -> None:
    class LikelihoodModel(BaseModel):
        prob: float = Field(description="Likelihood of conversion")

    questions = questions_from_model(LikelihoodModel)
    assert "prob" in questions
    assert isinstance(questions["prob"], Noul)
    assert questions["prob"].instructions == "Likelihood of conversion"

    response = make_system_one_response({"prob": NoulAnswer(noul=0.82)})
    result = parse_model(LikelihoodModel, response)
    assert result.prob == 0.82


def test_score_int_rounding_and_strict() -> None:
    class IntScoreModel(BaseModel):
        score: Annotated[int, ScoreConfig(levels=["low", "medium", "high"])]

    # 1.7 should round to 2
    res2 = parse_model(IntScoreModel, make_system_one_response({"score": ScoreAnswer(score=1.7, confidence=1.0, probabilities={}, legend={})}))
    assert res2.score == 2
    assert isinstance(res2.score, int)

    # 1.2 should round to 1
    res1 = parse_model(IntScoreModel, make_system_one_response({"score": ScoreAnswer(score=1.2, confidence=1.0, probabilities={}, legend={})}))
    assert res1.score == 1
    assert isinstance(res1.score, int)

    # Strict model should accept integer-rounded score without ValidationError
    class StrictScoreModel(BaseModel):
        model_config = ConfigDict(strict=True)
        score: Annotated[int, ScoreConfig(levels=["low", "medium", "high"])]

    strict_res = parse_model(StrictScoreModel, make_system_one_response({"score": ScoreAnswer(score=1.0, confidence=1.0, probabilities={}, legend={})}))
    assert strict_res.score == 1
    assert isinstance(strict_res.score, int)


class CustomChoice(ChoiceAnswer):
    pass


class CustomNoul(NoulAnswer):
    pass


class CustomScore(ScoreAnswer):
    pass


class SubclassModel(BaseModel):
    choice: Annotated[CustomChoice, ChoiceConfig(criteria=["apple", "banana"])]
    flag: CustomNoul
    rating: Annotated[CustomScore, ScoreConfig(levels=["bad", "good"])]


def test_answer_subclasses() -> None:
    questions = questions_from_model(SubclassModel)
    assert isinstance(questions["choice"], Choice)
    assert isinstance(questions["flag"], Noul)
    assert isinstance(questions["rating"], Score)

    response = make_system_one_response(
        {
            "choice": ChoiceAnswer(choice="apple", confidence=0.9, probabilities={"apple": 0.9, "banana": 0.1}),
            "flag": NoulAnswer(noul=0.8),
            "rating": ScoreAnswer(score=1.0, confidence=1.0, probabilities={0: 0.0, 1: 1.0}, legend={0: "bad", 1: "good"}),
        }
    )
    result = parse_model(SubclassModel, response)
    assert isinstance(result.choice, CustomChoice)
    assert result.choice.choice == "apple"
    assert isinstance(result.flag, CustomNoul)
    assert result.flag.noul == 0.8
    assert isinstance(result.rating, CustomScore)
    assert result.rating.score == 1.0


def test_choice_criteria_sequences() -> None:
    class SeqLiteralModel(BaseModel):
        tag: Annotated[Literal["x", "y"], ChoiceConfig(criteria=["x", "y"])]

    q1 = questions_from_model(SeqLiteralModel)
    assert isinstance(q1["tag"], Choice)
    assert set(q1["tag"].criteria.keys()) == {"x", "y"}

    class SeqStrModel(BaseModel):
        tag: Annotated[str, QuestionConfig(criteria=("alpha", "beta"))]

    q2 = questions_from_model(SeqStrModel)
    assert isinstance(q2["tag"], Choice)
    assert set(q2["tag"].criteria.keys()) == {"alpha", "beta"}

    # Incompatible sequence criteria
    class IncompatibleSeq(BaseModel):
        tag: Annotated[Literal["x", "y"], ChoiceConfig(criteria=["x", "z"])]

    with pytest.raises(TypeSafeError, match="incompatible Choice metadata"):
        questions_from_model(IncompatibleSeq)


def test_score_config_list_and_tuple_equality() -> None:
    sc = ScoreConfig(levels=["a", "b"], criteria=("a", "b"))
    assert sc.levels is not None and list(sc.levels) == ["a", "b"]
    assert sc.criteria is not None and list(sc.criteria) == ["a", "b"]

    sc2 = ScoreConfig(levels=("x", "y"), criteria=["x", "y"])
    assert sc2.levels is not None and list(sc2.levels) == ["x", "y"]
    assert sc2.criteria is not None and list(sc2.criteria) == ["x", "y"]


def test_score_boundary_epsilon() -> None:
    class SimpleScore(BaseModel):
        val: Annotated[ScoreAnswer, ScoreConfig(levels=["l1", "l2", "l3"])]

    # 2.0000000000000004 should be accepted within epsilon of max index 2
    res = parse_model(
        SimpleScore,
        make_system_one_response(
            {"val": ScoreAnswer(score=2.0000000000000004, confidence=1.0, probabilities={}, legend={})}
        ),
    )
    assert res.val.score == 2.0000000000000004


async def test_inferred_mode_with_pydantic_state_input(clients: ClientFactory) -> None:
    client = clients(
        lambda req: httpx2.Response(
            200,
            json={
                "model": "jev-latest",
                "usage": {"input_tokens": 10, "output_tokens": 2},
                "answers": {
                    "category": {"type": "choice", "choice": "billing", "confidence": 1.0, "probabilities": {"billing": 1.0}},
                    "urgent": {"type": "noul", "noul": 0.9},
                    "severity": {"type": "score", "score": 2.0, "confidence": 1.0, "legend": {"0": "l1", "1": "l2", "2": "l3"}, "probabilities": {"2": 1.0}},
                },
            },
        )
    )

    req = StateInputModel(message="charge error", code=500)
    system_one_fn = getattr(client, "system_one")  # noqa: B009
    if isinstance(client, AsyncTypeSafeClient):
        t1 = await client.system_one(input=req, response_model=Ticket)
        t2 = await system_one_fn(state=req, response_model=Ticket)
    else:
        t1 = client.system_one(input=req, response_model=Ticket)
        t2 = system_one_fn(state=req, response_model=Ticket)

    assert t1.category == "billing"
    assert t1.urgent is True
    assert t1.severity.score == 2.0
    assert t2.category == "billing"

