"""
FlowCoder: Ultra-Fast, Mobile-Friendly Hybrid Generative Framework
Based on Rectified Flow, Depthwise-Attention Hybrid CORE, and 1-8 Step Infiltration.
"""

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
