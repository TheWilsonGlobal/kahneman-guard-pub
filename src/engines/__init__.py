"""
KahnemanGuard Tier 1 Inference Engines.
Exports ModernBERT Dense, Discrete Diffusion, and Diffusion + RLCD engines.
"""

from src.engines.base_engine import BaseGuardEngine
from src.engines.onnx_dense_engine import ModernBERTDenseEngine
from src.engines.onnx_diffusion_engine import ModernBERTDiffusionEngine
from src.engines.onnx_diffusion_rlcd_engine import ModernBERTDiffusionRLCDEngine


def create_engine(variant: str = "dense", **kwargs) -> BaseGuardEngine:
    """Factory function for instantiating Tier 1 inference engines."""
    v = variant.lower().strip()
    if v in ("dense", "modernbert_dense"):
        return ModernBERTDenseEngine(**kwargs)
    elif v in ("diffusion", "diff", "modernbert_diffusion"):
        return ModernBERTDiffusionEngine(**kwargs)
    elif v in ("diffusion-rlcd", "diffusion_rlcd", "rlcd"):
        return ModernBERTDiffusionRLCDEngine(**kwargs)
    else:
        raise ValueError(f"Unknown guardrail engine variant: {variant}. Choose 'dense', 'diffusion', or 'diffusion-rlcd'.")


__all__ = [
    "BaseGuardEngine",
    "ModernBERTDenseEngine",
    "ModernBERTDiffusionEngine",
    "ModernBERTDiffusionRLCDEngine",
    "create_engine",
]
