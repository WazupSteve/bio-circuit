from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .circuit_supernodes import CircuitTaskset

# Verifiers v1 discovers exactly one Taskset subclass through __all__.
__all__ = ["CircuitTaskset"]


def __getattr__(name: str):
    # Toolsets are launched with ``python -m circuit_supernodes.<module>``. Keep
    # package import side-effect free so runpy does not pre-import that module.
    if name == "CircuitTaskset":
        from .circuit_supernodes import CircuitTaskset

        return CircuitTaskset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
