"""
Base Guardrail Inference Engine Interface.
Defines unified contract across Direct ONNX Runtime engines and microservice clients.
"""

from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel

from src.core.router import DecisionCategory, GuardrailDecision


class BaseGuardEngine(ABC):
    """Abstract base class for high-throughput Tier 1 guardrail inference engines."""

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Name of the engine architecture (e.g. ModernBERT-Dense, ModernBERT-Diffusion)."""
        pass

    @property
    @abstractmethod
    def temperature(self) -> float:
        """Calibrated post-hoc temperature parameter."""
        pass

    @property
    @abstractmethod
    def is_local_onnx(self) -> bool:
        """True if running native ONNX Runtime graph locally."""
        pass

    @abstractmethod
    def predict(self, prompt: str) -> GuardrailDecision:
        """Synchronous prompt safety evaluation."""
        pass

    def predict_batch(self, prompts: list[str]) -> list[GuardrailDecision]:
        """Batch evaluation for maximum throughput."""
        return [self.predict(p) for p in prompts]
