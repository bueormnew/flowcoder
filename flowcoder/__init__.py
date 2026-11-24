from __future__ import annotations
"""
FlowCoder: Ultra-Fast, Mobile-Friendly Hybrid Generative Framework
Based on Rectified Flow, Depthwise-Attention Hybrid CORE, and 1-8 Step Infiltration.
"""

# Patch de compatibilidad para PyTorch en entornos con versiones dispares (ej. Kaggle / Colab)
import torch
try:
    import torch._utils
    if not hasattr(torch._utils, "_chunk_or_narrow_cat"):
        def _chunk_or_narrow_cat(tensor, num_chunks, narrow_dim, cat_dim=0):
            return torch.cat(torch.chunk(tensor, num_chunks, dim=narrow_dim), dim=cat_dim)
        torch._utils._chunk_or_narrow_cat = _chunk_or_narrow_cat
except Exception:
    pass

from .config import (
    CoreConfig,
    PipelineConfig,
    TrainingConfig,
    LoRAConfig,
    EndToEndFinetuneConfig,
)
from .pipeline import FlowCoderPipeline

__version__ = "0.1.0"

__all__ = [
    "CoreConfig",
    "PipelineConfig",
    "TrainingConfig",
    "LoRAConfig",
    "EndToEndFinetuneConfig",
    "FlowCoderPipeline",
]
