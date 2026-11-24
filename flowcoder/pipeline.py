from __future__ import annotations
from typing import List, Optional, Union, Dict, Any
from pathlib import Path
import json
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

from .config import PipelineConfig, CoreConfig, LoRAConfig
from .models.core import MobileCoreDenoiser
from .models.encoders import BaseTextEncoder, HFTextEncoder, CustomTextEncoder
from .models.vaes import BaseVAE, DiffusersVAE, TinyVAE, CustomVAE
from .models.lora import inject_lora, merge_lora, load_lora_weights
from .solvers.flow_solver import FlowSolver


class FlowCoderPipeline:
    """
    Gestor unificado y pipeline de inferencia para arquitecturas FlowCoder.
    Coordina Text Encoder, CORE híbrido y VAE, soportando modelos preentrenados y personalizados.
    """
    def __init__(
        self,
        core: MobileCoreDenoiser,
        text_encoder: BaseTextEncoder,
        vae: BaseVAE,
        config: Optional[PipelineConfig] = None,
    ):
        self.core = core
        self.text_encoder = text_encoder
        self.vae = vae
        self.config = config or PipelineConfig(core=core.config)
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def to(self, device: Union[str, torch.device], dtype: Optional[torch.dtype] = None) -> "FlowCoderPipeline":
        """Mueve todo el pipeline al dispositivo y precisión deseados."""
        self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
            self.core = self.core.to(device=self.device, dtype=dtype)
            self.text_encoder = self.text_encoder.to(device=self.device, dtype=dtype)
            self.vae = self.vae.to(device=self.device, dtype=dtype)
        else:
            self.core = self.core.to(self.device)
            self.text_encoder = self.text_encoder.to(self.device)
            self.vae = self.vae.to(self.device)
        return self

    @classmethod
    def create(
        cls,
        core_config: Optional[CoreConfig] = None,
        text_encoder: Union[str, BaseTextEncoder] = "google/flan-t5-small",
        vae: Union[str, BaseVAE] = "tiny", # "tiny" para TinyVAE (móvil) o nombre de HuggingFace
        resolution: int = 512,
    ) -> "FlowCoderPipeline":
        """
        Crea una instancia nueva de FlowCoder con componentes configurables.
        """
        # 1. Configurar Text Encoder
        if isinstance(text_encoder, str):
            enc = HFTextEncoder(text_encoder)
        else:
            enc = text_encoder

        # 2. Configurar VAE
        if isinstance(vae, str):
            if vae.lower() in ["tiny", "taesd"]:
                v = TinyVAE()
            else:
                v = DiffusersVAE(vae)
        else:
            v = vae

        # 3. Configurar CORE
        if core_config is None:
            core_config = CoreConfig(
                in_channels=v.latent_channels,
                out_channels=v.latent_channels,
                txt_dim=enc.output_dim,
            )
        else:
            core_config.in_channels = v.latent_channels
            core_config.out_channels = v.latent_channels
            core_config.txt_dim = enc.output_dim

        core_model = MobileCoreDenoiser(core_config)
        pipe_cfg = PipelineConfig(
            core=core_config,
            resolution=resolution,
        )
        return cls(core=core_model, text_encoder=enc, vae=v, config=pipe_cfg)

    @torch.no_grad()
    def generate(
        self,
        prompts: Union[str, List[str]],
        steps: int = 4,
        cfg_scale: float = 3.0,
        solver: str = "euler",            # "euler" o "heun"
        seed: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        one_pass_guidance: bool = True,
        output_type: str = "pil",         # "pil", "np", "pt"
    ) -> List[Image.Image] | np.ndarray | torch.Tensor:
        """
        Genera imágenes en 1 a 8 pasos a partir de prompts de texto.
        """
        self.core.eval()
        self.text_encoder.eval()
        self.vae.eval()

        if isinstance(prompts, str):
            prompts = [prompts]
        B = len(prompts)

        h = height or self.config.resolution
        w = width or self.config.resolution
        ds = self.vae.downsampling_factor

        lat_h = h // ds
        lat_w = w // ds

        # 1. Generador de ruido reproducible
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Ruido inicial z_1 ~ N(0, I) en espacio latente
        x_init = torch.randn(
            (B, self.vae.latent_channels, lat_h, lat_w),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

        # 2. Codificación de texto (se ejecuta una única vez)
        txt_ctx, txt_mask = self.text_encoder(prompts, device=self.device)
        txt_ctx = txt_ctx.to(dtype=self.dtype)

        # 3. Solucionador de Rectified Flow
        if solver.lower() == "heun":
            latents = FlowSolver.sample_heun(
                model=self.core,
                latents=x_init,
                txt_ctx=txt_ctx,
                txt_mask=txt_mask,
                steps=steps,
                cfg_scale=cfg_scale,
                one_pass_guidance=one_pass_guidance,
            )
        else:
            latents = FlowSolver.sample_euler(
                model=self.core,
                latents=x_init,
                txt_ctx=txt_ctx,
                txt_mask=txt_mask,
                steps=steps,
                cfg_scale=cfg_scale,
                one_pass_guidance=one_pass_guidance,
            )

        # 4. Decodificación de latentes a RGB
        images_tensor = self.vae.decode(latents.to(self.device))
        images_tensor = ((images_tensor.clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(0.0, 1.0)

        if output_type == "pt":
            return images_tensor

        images_np = (images_tensor.permute(0, 2, 3, 1).cpu().float().numpy() * 255.0).round().astype(np.uint8)
        if output_type == "np":
            return images_np

        return [Image.fromarray(img) for img in images_np]

    def load_lora(self, lora_path: Path | str, lora_config: Optional[LoRAConfig] = None) -> None:
        """Inyecta y carga pesos LoRA en el CORE."""
        cfg = lora_config or LoRAConfig()
        inject_lora(self.core, cfg)
        load_lora_weights(self.core, lora_path)

    def merge_lora_weights(self) -> None:
        """Fusiona los pesos LoRA en el modelo base para máxima velocidad de inferencia."""
        merge_lora(self.core)

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """Exporta todo el paquete del modelo con configuración y pesos."""
        p = Path(save_directory)
        p.mkdir(parents=True, exist_ok=True)

        # Guardar configuración
        self.config.save_json(p / "pipeline_config.json")

        # Guardar pesos del CORE
        core_state = {k: v.contiguous() for k, v in self.core.state_dict().items()}
        try:
            from safetensors.torch import save_file
            save_file(core_state, str(p / "core.safetensors"))
        except ImportError:
            torch.save(core_state, str(p / "core.pt"))

    @classmethod
    def from_pretrained(
        cls,
        load_directory: Union[str, Path],
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "FlowCoderPipeline":
        """Carga un pipeline completo guardado previamente."""
        p = Path(load_directory)
        cfg = PipelineConfig.from_json(p / "pipeline_config.json")
        pipe = cls.create(
            core_config=cfg.core,
            text_encoder=cfg.encoder_name_or_path,
            vae=cfg.vae_name_or_path,
            resolution=cfg.resolution,
        )

        weights_file = p / "core.safetensors"
        if weights_file.exists():
            from safetensors.torch import load_file
            state = load_file(str(weights_file), device=device)
        else:
            state = torch.load(str(p / "core.pt"), map_location=device)

        pipe.core.load_state_dict(state)
        return pipe.to(device=device, dtype=dtype)
