from __future__ import annotations
import math
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from pathlib import Path

from ..config import LoRAConfig


class LoRALinear(nn.Module):
    """Capa lineal con adaptador Low-Rank Adaptation (LoRA)."""
    def __init__(self, base_layer: nn.Linear, r: int = 16, alpha: float = 32.0, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # Inicialización: A con kaiming_uniform, B con ceros (para iniciar como identidad)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Congelar la capa base
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return base_out + lora_out

    def merge(self) -> None:
        """Fusiona los pesos de LoRA en la capa base para inferencia de costo cero."""
        with torch.no_grad():
            self.base_layer.weight.data += (self.lora_B @ self.lora_A) * self.scaling
        self.merged = True


class LoRAConv2d(nn.Module):
    """Adaptador LoRA para convoluciones 2D (Pointwise 1x1 o Depthwise)."""
    def __init__(self, base_layer: nn.Conv2d, r: int = 16, alpha: float = 32.0, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0.0 else nn.Identity()

        in_channels = base_layer.in_channels
        out_channels = base_layer.out_channels
        kernel_size = base_layer.kernel_size

        self.lora_A = nn.Parameter(torch.zeros(r, in_channels, *kernel_size))
        self.lora_B = nn.Parameter(torch.zeros(out_channels, r, 1, 1))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        # Convolución con A y luego con B
        lora_h = F.conv2d(
            self.dropout(x),
            self.lora_A,
            stride=self.base_layer.stride,
            padding=self.base_layer.padding,
            dilation=self.base_layer.dilation,
            groups=1,
        )
        lora_out = F.conv2d(lora_h, self.lora_B) * self.scaling
        return base_out + lora_out


def inject_lora(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """
    Inyecta automáticamente adaptadores LoRA en todas las capas coincidentes
    con target_modules y congela el resto de los parámetros.
    """
    for param in model.parameters():
        param.requires_grad = False

    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if any(target in child_name for target in config.target_modules):
                if isinstance(child, nn.Linear):
                    lora_layer = LoRALinear(child, r=config.r, alpha=config.alpha, dropout=config.dropout)
                    setattr(module, child_name, lora_layer)
                elif isinstance(child, nn.Conv2d) and child.groups == 1:
                    lora_layer = LoRAConv2d(child, r=config.r, alpha=config.alpha, dropout=config.dropout)
                    setattr(module, child_name, lora_layer)
    return model


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extrae únicamente los pesos entrenables de los adaptadores LoRA."""
    return {k: v.contiguous() for k, v in model.state_dict().items() if "lora_" in k}


def merge_lora(model: nn.Module) -> nn.Module:
    """Recorre el modelo y fusiona todos los pesos LoRA con las capas base."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
    return model


def save_lora_weights(model: nn.Module, path: Path | str) -> None:
    """Guarda los pesos LoRA en formato PyTorch o Safetensors."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    state = get_lora_state_dict(model)
    try:
        from safetensors.torch import save_file
        save_file(state, str(p))
    except ImportError:
        torch.save(state, str(p))


def load_lora_weights(model: nn.Module, path: Path | str) -> nn.Module:
    """Carga pesos LoRA en un modelo que ya tiene adaptadores inyectados."""
    p = Path(path)
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(p))
    else:
        state = torch.load(str(p), map_location="cpu")
    model.load_state_dict(state, strict=False)
    return model
