from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class AdapterError(RuntimeError):
    """Raised when a generation task cannot be executed safely."""


class GenerationAdapter(ABC):
    """Minimal contract shared by every generation method."""

    method_id: str

    @abstractmethod
    def load(self) -> None:
        """Load model components once before executing one or more tasks."""

    @abstractmethod
    def generate(self, task: dict[str, Any], output_path: Path) -> dict[str, Any]:
        """Generate one image and return method-specific metadata."""

    def close(self) -> None:
        """Release resources after the run; adapters may override this."""

