from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import torch
import torch.nn as nn

from .._compat import apply_torch_compat_patches


class BaseTextEncoder(nn.Module, ABC):
    """Interfaz base para cualquier Text Encoder en FlowCoder."""

    @abstractmethod
    def forward(self, texts: List[str] | torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retorna:
          embeddings: (B, L, txt_dim)
          attention_mask: (B, L) booleano (True = atender)
        """
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensión del espacio latente de texto."""
        pass


class HFTextEncoder(BaseTextEncoder):
    """
    Wrapper compatible con modelos de Hugging Face (T5, CLIP, MobileCLIP, RoBERTa, etc.).
    Soporta cualquier encoder preentrenado estándar.
    """
    def __init__(self, model_name_or_path: str = "google/flan-t5-small", max_length: int = 77):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self._dim = None

        # Asegurar parches de compatibilidad para PyTorch / transformers distributed utils
        apply_torch_compat_patches()

        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Detectar si es un encoder T5 o CLIP o modelo general
        if "t5" in model_name_or_path.lower():
            try:
                from transformers import T5EncoderModel
                self.model = T5EncoderModel.from_pretrained(model_name_or_path)
            except (AttributeError, ImportError):
                # Re-aplicar parches por si transformers se cargó parcialmente y usar AutoModel
                apply_torch_compat_patches()
                self.model = AutoModel.from_pretrained(model_name_or_path)
                if hasattr(self.model, "encoder") and not hasattr(self.model, "last_hidden_state"):
                    # T5Model completo: conservar sólo el encoder si está disponible
                    self.model = self.model.encoder
            self._dim = self.model.config.d_model
        elif "clip" in model_name_or_path.lower():
            from transformers import CLIPTextModel
            self.model = CLIPTextModel.from_pretrained(model_name_or_path)
            self._dim = self.model.config.hidden_size
        else:
            self.model = AutoModel.from_pretrained(model_name_or_path)
            self._dim = getattr(self.model.config, "hidden_size", getattr(self.model.config, "d_model", 512))

    @property
    def output_dim(self) -> int:
        return self._dim

    def forward(self, texts: List[str] | Tuple[str, ...] | torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(texts, (list, tuple)):
            tokens = self.tokenizer(
                list(texts),
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            input_ids = tokens.input_ids
            attention_mask = tokens.attention_mask
        else:
            input_ids = texts.to(device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        return last_hidden, attention_mask.bool()


class CustomTextEncoder(BaseTextEncoder):
    """
    Permite integrar un modelo propio o personalizado creado por el usuario.
    Recibe una función o nn.Module propio y su dimensión de salida.
    """
    def __init__(self, encoder_module: nn.Module, output_dim: int, tokenizer_fn=None, max_length: int = 77):
        super().__init__()
        self.encoder = encoder_module
        self._output_dim = output_dim
        self.tokenizer_fn = tokenizer_fn
        self.max_length = max_length

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, texts: List[str] | Tuple[str, ...] | torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.tokenizer_fn is not None and isinstance(texts, (list, tuple)):
            tokens, mask = self.tokenizer_fn(list(texts), max_length=self.max_length, device=device)
            embs = self.encoder(tokens)
            return embs, mask
        elif isinstance(texts, torch.Tensor):
            mask = torch.ones(texts.shape[0], texts.shape[1], dtype=torch.bool, device=device)
            return self.encoder(texts.to(device)), mask
        else:
            return self.encoder(texts)
