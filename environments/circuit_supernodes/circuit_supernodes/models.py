"""Declarative Pydantic schemas shared by the circuit runtime."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    StrictInt,
    StringConstraints,
    field_validator,
    model_validator,
)

NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
NonBlankStr = Annotated[str, StringConstraints(pattern=r"\S")]
CoordinateTuple = tuple[NonNegativeInt, NonNegativeInt, NonNegativeInt]
Intervention = tuple[NonNegativeInt, NonNegativeInt, NonNegativeInt, FiniteFloat]


class StrictModel(BaseModel):
    """Strict project models that reject unknown fields."""

    model_config = ConfigDict(strict=True, extra="forbid")


class FrozenStrictModel(StrictModel):
    """Compatibility base for immutable repository-local builder models."""

    model_config = ConfigDict(frozen=True)


GroupName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
StableFeatureId = Annotated[str, StringConstraints(pattern=r"^\d+_\d+$")]


def _nonzero_weight(weight: float) -> float:
    if weight == 0.0:
        raise ValueError("intervention_weight must be nonzero")
    return weight


BoundedWeight = Annotated[
    FiniteFloat, Field(ge=-8.0, le=8.0), AfterValidator(_nonzero_weight)
]
SupernodeWeights = Annotated[
    dict[StableFeatureId, BoundedWeight], Field(min_length=1, max_length=60)
]


class Circuit(StrictModel):
    """Canonical named groups of stable feature IDs and intervention weights."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    supernodes: dict[GroupName, SupernodeWeights] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def unique_groups_and_features(self) -> Self:
        names = [name.casefold() for name in self.supernodes]
        if len(names) != len(set(names)):
            raise ValueError("supernode names must be unique")
        features = [
            feature for supernode in self.supernodes.values() for feature in supernode
        ]
        if len(features) > 60:
            raise ValueError("use at most 60 intervention nodes")
        if len(features) != len(set(features)):
            raise ValueError("duplicate stable FEATURE id")
        return self


class TracerMetadata(StrictModel):
    """Circuit Tracer metadata consumed by this package."""

    model_config = ConfigDict(strict=True, extra="ignore")
    prompt_tokens: list[str] = Field(default_factory=list)


class TracerNode(StrictModel):
    """Canonical consumed subset of an exported Circuit Tracer node."""

    model_config = ConfigDict(strict=True, extra="ignore")
    node_id: str
    feature_type: str = ""
    ctx_idx: int = 0
    layer: int | str = 0
    influence: float | None = None
    clerp: str = ""
    label: str = ""
    activation: float | None = None
    token_prob: float = 0.0


class TracerLink(StrictModel):
    """Canonical consumed subset of an exported Circuit Tracer link."""

    model_config = ConfigDict(strict=True, extra="ignore")
    source: str
    target: str
    weight: float = 0.0


class TracerGraphPayload(StrictModel):
    """Canonical Circuit Tracer graph export."""

    model_config = ConfigDict(strict=True, extra="ignore")
    metadata: TracerMetadata = Field(default_factory=TracerMetadata)
    nodes: list[TracerNode] = Field(default_factory=list)
    links: list[TracerLink] = Field(default_factory=list)


class DatasetExample(StrictModel):
    """Runtime-minimal prompt, target, graph, and optional corruption."""

    prompt: str = Field(min_length=1)
    target_token: str = Field(min_length=1)
    graph: TracerGraphPayload
    corruption: list[Intervention] = Field(default_factory=list)

    @field_validator("corruption", mode="before")
    @classmethod
    def json_interventions(cls, value: object) -> object:
        if isinstance(value, list):
            return [tuple(item) if isinstance(item, list) else item for item in value]
        return value


class CircuitDatasetRow(StrictModel):
    """Runtime-minimal three-cohort row used by every circuit taskset."""

    family: str = Field(min_length=1)
    goal: str | None = None
    desired_answer: str | None = None
    example_prompts: list[DatasetExample]
    heldout_prompts: list[DatasetExample]
    control_prompts: list[DatasetExample]


class SamplingRequest(StrictModel):
    """Validated input to the bounded remote subject sampler."""

    prompt: NonBlankStr
    possible_answers: list[
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    ] = Field(min_length=1)
    max_new_tokens: int = Field(default=50, ge=1, le=50)
    debug: bool = False


class SamplingResult(StrictModel):
    """Outcome returned by bounded remote answer generation."""

    status: Literal["matched", "exhausted", "error"]
    generated_text: str = ""
    matched_answer: str | None = None
    provided_prompt: str | None = None
    target_token: str | None = None
    generated_tokens: int | None = Field(default=None, ge=0)
    error: str | None = None
