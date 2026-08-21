"""Unified Modal backend, split by execution responsibility."""

from .deployment import app
from .service import Attributor

__all__ = ["Attributor", "app"]
