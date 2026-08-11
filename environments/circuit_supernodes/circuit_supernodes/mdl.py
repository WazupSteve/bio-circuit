"""First-principles minimum-description-length scoring for causal circuits.

The GPU backend produces a proposal-independent :class:`ResponseBank`.  This
module is deliberately pure: it decodes a proposal into a block-constant causal
operator, measures held-out distortion, charges the literal circuit code, and
maps normalized net bit saving to a strictly positive exponential reward.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

OUTPUT = "output"
MAX_GROUPS = 8
BIT_SCALE = 2.0 * math.log(2.0)

Matrix = tuple[tuple[float, ...], ...]
Tensor = tuple[Matrix, ...]


class InvalidResponseBank(ValueError):
    """The frozen calibration/validation data does not satisfy its contract."""


class UnscorableResponseBank(ValueError):
    """The null response has no positive output-relevant energy."""


def exponential_reward(net_reward: float) -> float:
    """Map signed normalized MDL value monotonically into ``(0, 1]``.

    A break-even circuit has reward ``0.5``; net-saving circuits score above
    ``0.5`` and net-cost circuits score below it.  The smallest positive float
    prevents extreme finite costs from underflowing to the invalid-proposal
    sentinel ``0.0``.
    """

    reward = math.exp2(net_reward - 1.0)
    return max(math.ulp(0.0), reward)


def _finite_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidResponseBank(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise InvalidResponseBank(f"{label} must be finite")
    return number


def _matrix(raw: Any, rows: int, cols: int, label: str) -> Matrix:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise InvalidResponseBank(f"{label} must be a matrix")
    if len(raw) != rows:
        raise InvalidResponseBank(f"{label} has {len(raw)} rows; expected {rows}")
    result: list[tuple[float, ...]] = []
    for i, row in enumerate(raw):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise InvalidResponseBank(f"{label}[{i}] must be a row")
        if len(row) != cols:
            raise InvalidResponseBank(
                f"{label}[{i}] has {len(row)} columns; expected {cols}"
            )
        result.append(
            tuple(
                _finite_float(value, f"{label}[{i}][{j}]")
                for j, value in enumerate(row)
            )
        )
    return tuple(result)


def _tensor(raw: Any, rows: int, cols: int, label: str) -> Tensor:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise InvalidResponseBank(f"{label} must contain at least one matrix")
    return tuple(
        _matrix(matrix, rows, cols, f"{label}[{sample}]")
        for sample, matrix in enumerate(raw)
    )


@dataclass(frozen=True)
class ResponseBank:
    """Frozen, proposal-independent causal responses.

    Every response matrix has ``N`` source-intervention rows and ``N + 1``
    destination columns.  The final column is target log-odds.  Calibration
    matrices fit block values; disjoint validation matrices alone determine the
    reward.  Scales and output-relevance weights are fixed before a proposal is
    decoded.
    """

    feature_ids: tuple[str, ...]
    calibration: Tensor
    validation: Tensor
    noise_scales: Matrix
    column_weights: tuple[float, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ResponseBank:
        if not isinstance(data, Mapping):
            raise InvalidResponseBank("response bank must be an object")
        raw_ids = data.get("feature_ids")
        if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes)):
            raise InvalidResponseBank("feature_ids must be a list")
        feature_ids = tuple(str(value) for value in raw_ids)
        if not feature_ids:
            raise InvalidResponseBank("feature_ids must not be empty")
        if len(set(feature_ids)) != len(feature_ids):
            raise InvalidResponseBank("feature_ids must be unique")

        rows, cols = len(feature_ids), len(feature_ids) + 1
        calibration = _tensor(data.get("calibration"), rows, cols, "calibration")
        validation = _tensor(data.get("validation"), rows, cols, "validation")

        raw_scales = data.get("noise_scales")
        if (
            isinstance(raw_scales, Sequence)
            and not isinstance(raw_scales, (str, bytes))
            and len(raw_scales) == cols
            and all(not isinstance(value, Sequence) for value in raw_scales)
        ):
            scale_row = tuple(
                _finite_float(value, f"noise_scales[{j}]")
                for j, value in enumerate(raw_scales)
            )
            noise_scales = tuple(scale_row for _ in range(rows))
        else:
            noise_scales = _matrix(raw_scales, rows, cols, "noise_scales")
        if any(scale <= 0.0 for row in noise_scales for scale in row):
            raise InvalidResponseBank("every noise scale must be positive")

        raw_weights = data.get("column_weights")
        if not isinstance(raw_weights, Sequence) or isinstance(
            raw_weights, (str, bytes)
        ):
            raise InvalidResponseBank("column_weights must be a list")
        if len(raw_weights) != cols:
            raise InvalidResponseBank(
                f"column_weights has {len(raw_weights)} entries; expected {cols}"
            )
        column_weights = tuple(
            _finite_float(value, f"column_weights[{j}]")
            for j, value in enumerate(raw_weights)
        )
        if any(weight < 0.0 for weight in column_weights):
            raise InvalidResponseBank("column weights must be nonnegative")
        if not any(weight > 0.0 for weight in column_weights):
            raise InvalidResponseBank("at least one column weight must be positive")

        metadata = data.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise InvalidResponseBank("metadata must be an object")
        return cls(
            feature_ids=feature_ids,
            calibration=calibration,
            validation=validation,
            noise_scales=noise_scales,
            column_weights=column_weights,
            metadata=dict(metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_ids": list(self.feature_ids),
            "calibration": [
                [list(row) for row in matrix] for matrix in self.calibration
            ],
            "validation": [[list(row) for row in matrix] for matrix in self.validation],
            "noise_scales": [list(row) for row in self.noise_scales],
            "column_weights": list(self.column_weights),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MDLScore:
    reward: float
    net_reward: float
    null_error: float
    circuit_error: float
    description_bits: int
    predictive_bits: float
    net_bits: float
    validation_matrices: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "reward": self.reward,
            "net_reward": self.net_reward,
            "null_error": self.null_error,
            "circuit_error": self.circuit_error,
            "distortion_ratio": self.circuit_error / self.null_error,
            "description_bits": self.description_bits,
            "description_error_cost": BIT_SCALE * self.description_bits,
            "predictive_bits": self.predictive_bits,
            "net_bits": self.net_bits,
            "bit_scale": BIT_SCALE,
            "validation_matrices": self.validation_matrices,
        }


def ceil_log2_cardinality(cardinality: int) -> int:
    """Bits in a fixed-width field containing ``cardinality`` possible values."""

    if cardinality < 1:
        raise ValueError("cardinality must be positive")
    return (cardinality - 1).bit_length()


def literal_description_bits(
    *,
    universe_size: int,
    selected_count: int,
    group_count: int,
    edge_count: int,
    max_groups: int = MAX_GROUPS,
) -> int:
    """Length of the canonical literal prefix circuit code from the proof.

    Counts are self-delimiting fixed-width fields.  Feature identifiers and
    group assignments are emitted literally in sorted feature order; every edge
    emits two literal endpoint labels.  Names and fitted block values are shared
    side information and are not encoded.
    """

    if universe_size < 1:
        raise ValueError("universe_size must be positive")
    if not 0 <= selected_count <= universe_size:
        raise ValueError("selected_count must lie inside the feature universe")
    if not 1 <= group_count <= max_groups:
        raise ValueError(f"group_count must lie in 1..{max_groups}")
    possible_edges = group_count * (group_count - 1) + group_count
    if not 0 <= edge_count <= possible_edges:
        raise ValueError("edge_count exceeds the literal endpoint universe")

    b_n = ceil_log2_cardinality(universe_size + 1)
    b_node = ceil_log2_cardinality(universe_size)
    b_k = ceil_log2_cardinality(max_groups + 1)
    b_group = ceil_log2_cardinality(group_count)
    b_m = ceil_log2_cardinality(possible_edges + 1)
    b_endpoint = ceil_log2_cardinality(group_count + 1)
    return (
        b_n
        + b_k
        + selected_count * (b_node + b_group)
        + b_m
        + 2 * edge_count * b_endpoint
    )


def _proposal_membership(
    supernodes: Mapping[str, Sequence[str]], bank: ResponseBank
) -> tuple[dict[str, str], dict[str, tuple[int, ...]]]:
    index = {feature_id: i for i, feature_id in enumerate(bank.feature_ids)}
    membership: dict[str, str] = {}
    member_indices: dict[str, tuple[int, ...]] = {}
    for name, members in supernodes.items():
        if len(members) < 2:
            raise ValueError(f"{name!r} needs two members for leave-one-member-out fit")
        indices: list[int] = []
        for feature_id in members:
            if feature_id not in index:
                raise ValueError(f"{feature_id!r} is outside the response bank")
            if feature_id in membership:
                raise ValueError(f"{feature_id!r} occurs in multiple supernodes")
            membership[feature_id] = name
            indices.append(index[feature_id])
        member_indices[name] = tuple(indices)
    return membership, member_indices


def score_circuit(
    bank: ResponseBank,
    supernodes: Mapping[str, Sequence[str]],
    closure: Mapping[str, set[str]],
    *,
    edge_count: int,
    max_groups: int = MAX_GROUPS,
) -> MDLScore:
    """Score a circuit by held-out causal compression.

    For every validation coordinate, the permitted block value is fitted from
    all calibration matrices and every *other* member of the source supernode.
    Unsubmitted rows, unsubmitted destinations, and unreachable blocks predict
    zero.  This makes omission and random grouping visible in a fixed universe.
    """

    if not supernodes:
        raise ValueError("a circuit must contain at least one supernode")
    if set(closure) != set(supernodes):
        raise ValueError("closure must contain exactly the submitted supernodes")
    for source, reached in closure.items():
        unknown = set(reached) - set(supernodes) - {OUTPUT}
        if unknown:
            raise ValueError(
                f"{source!r} reaches unknown destinations: {sorted(unknown)}"
            )

    membership, member_indices = _proposal_membership(supernodes, bank)
    group_for_index = {
        i: membership[feature_id]
        for i, feature_id in enumerate(bank.feature_ids)
        if feature_id in membership
    }
    selected_count = len(membership)
    description_bits = literal_description_bits(
        universe_size=len(bank.feature_ids),
        selected_count=selected_count,
        group_count=len(supernodes),
        edge_count=edge_count,
        max_groups=max_groups,
    )

    block_cache: dict[tuple[str, str, int], float] = {}

    def fitted_value(source_group: str, destination: str, held_out: int) -> float:
        key = (source_group, destination, held_out)
        if key in block_cache:
            return block_cache[key]
        sources = [i for i in member_indices[source_group] if i != held_out]
        if not sources:
            raise ValueError(
                f"{source_group!r} has no calibration member after holding one out"
            )
        if destination == OUTPUT:
            destinations = (len(bank.feature_ids),)
        else:
            destinations = member_indices[destination]
        values = [
            matrix[source][target]
            for matrix in bank.calibration
            for source in sources
            for target in destinations
        ]
        estimate = math.fsum(values) / len(values)
        block_cache[key] = estimate
        return estimate

    null_terms: list[float] = []
    circuit_terms: list[float] = []
    output_index = len(bank.feature_ids)
    for matrix in bank.validation:
        for source in range(len(bank.feature_ids)):
            source_group = group_for_index.get(source)
            reached = closure[source_group] if source_group is not None else set()
            for destination in range(output_index + 1):
                truth = matrix[source][destination]
                prediction = 0.0
                if source_group is not None:
                    if destination == output_index:
                        if OUTPUT in reached:
                            prediction = fitted_value(source_group, OUTPUT, source)
                    else:
                        destination_group = group_for_index.get(destination)
                        if (
                            destination_group is not None
                            and destination_group in reached
                        ):
                            prediction = fitted_value(
                                source_group, destination_group, source
                            )
                multiplier = bank.column_weights[destination]
                scale = bank.noise_scales[source][destination]
                null_terms.append(multiplier * (truth / scale) ** 2)
                circuit_terms.append(multiplier * ((truth - prediction) / scale) ** 2)

    null_error = math.fsum(null_terms)
    if not math.isfinite(null_error):
        raise UnscorableResponseBank("the whitened null error is not finite")
    if not null_error > 0.0:
        raise UnscorableResponseBank(
            "the fixed validation tensor has no output-relevant causal energy"
        )
    circuit_error = math.fsum(circuit_terms)
    if not math.isfinite(circuit_error):
        raise UnscorableResponseBank("the whitened circuit error is not finite")
    predictive_bits = (null_error - circuit_error) / BIT_SCALE
    net_bits = predictive_bits - description_bits
    net_reward = (
        null_error - circuit_error - BIT_SCALE * description_bits
    ) / null_error
    return MDLScore(
        reward=exponential_reward(net_reward),
        net_reward=net_reward,
        null_error=null_error,
        circuit_error=circuit_error,
        description_bits=description_bits,
        predictive_bits=predictive_bits,
        net_bits=net_bits,
        validation_matrices=len(bank.validation),
    )
