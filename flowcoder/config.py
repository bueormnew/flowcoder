from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import json
from pathlib import Path


@dataclass
class CoreConfig:
    """Configuración del modelo CORE híbrido (Depthwise + Attention + AdaLN)."""
    in_channels: int = 4
    out_channels: int = 4
    patch_size: int = 2              # 2x2 patchify para resoluciones 512x512 y 1024x1024
    dim: int = 384                   # Dimensión oculta del CORE
    depth: int = 12                  # Número total de bloques híbridos
    heads: int = 6                   # Cabezas de atención (dim // heads = 64)
    txt_dim: int = 512               # Dimensión de entrada de embeddings de texto
    time_embed_dim: int = 256        # Dimensión de embedding sinusoidal de tiempo
    cross_attn_interval: int = 2     # Cross-attention cada N bloques para optimizar móvil
    self_attn_layers: List[int] = field(default_factory=lambda: [8, 9, 10, 11]) # Self-attention sólo en capas profundas
    mlp_ratio: float = 4.0           # Expansión de FFN pointwise
    dw_kernel_size: int = 7          # Kernel de convolución depthwise (estilo ConvNeXt)
    use_guidance_embed: bool = True  # Acondicionamiento de escala de guía para 1-pass CFG

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CoreConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PipelineConfig:
    """Configuración general del Pipeline y Model Manager."""
    core: CoreConfig = field(default_factory=CoreConfig)
    encoder_type: str = "hf"         # "hf", "custom"
    encoder_name_or_path: str = "google/flan-t5-small"
    vae_type: str = "diffusers"      # "diffusers", "taesd", "custom"
    vae_name_or_path: str = "stabilityai/sd-vae-ft-mse"
    vae_scale_factor: float = 0.18215
    resolution: int = 512
    max_text_len: int = 77

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["core"] = self.core.to_dict()
        return d

    def save_json(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Path | str) -> "PipelineConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        core_cfg = CoreConfig.from_dict(data.pop("core", {}))
        return cls(core=core_cfg, **data)


@dataclass
class TrainingConfig:
    """Configuración para preentrenamiento o finetuning del CORE."""
    lr: float = 2e-4
    batch_size: int = 16
    epochs: int = 50
    weight_decay: float = 0.01
    warmup_steps: int = 500
    cfg_dropout: float = 0.1
    time_sampling: str = "uniform"   # "uniform", "logit_normal", "beta"
    mixed_precision: str = "fp16"    # "no", "fp16", "bf16"
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    ema_decay: float = 0.999
    save_every_epochs: int = 5


@dataclass
class LoRAConfig:
    """Configuración para Low-Rank Adaptation (LoRA) sobre el CORE o Encoder."""
    r: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "to_q", "to_kv", "proj_out", "pwconv1", "pwconv2"
    ])
    bias: str = "none"


@dataclass
class EndToEndFinetuneConfig:
    """
    Configuración para Refinamiento Completo Extremo a Extremo:
    Permite afinar ENCODER + CORE + VAE de manera armónica con diferentes hiperparámetros.
    """
    lr_encoder: float = 1e-5
    lr_core: float = 1e-4
    lr_vae: float = 5e-6
    weight_decay: float = 0.01
    epochs: int = 20
    batch_size: int = 8
    flow_loss_weight: float = 1.0
    recon_loss_weight: float = 0.5
    perceptual_loss_weight: float = 0.2
    freeze_encoder: bool = False
    freeze_vae: bool = False
    use_gradient_checkpointing: bool = True
