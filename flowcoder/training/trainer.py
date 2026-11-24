from typing import Optional, Dict, Any, List
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import TrainingConfig, LoRAConfig, EndToEndFinetuneConfig
from ..pipeline import FlowCoderPipeline
from ..models.lora import inject_lora, save_lora_weights, get_lora_state_dict
from .losses import FlowMatchingLoss, EndToEndReconstructionLoss
from .caching import LatentTextCacheDataset


class FlowTrainer:
    """
    Entrenador optimizado para el CORE mediante Rectified Flow y embeddings en caché.
    Alcanza alta velocidad al no re-ejecutar el VAE ni el Text Encoder en cada época.
    """
    def __init__(
        self,
        pipeline: FlowCoderPipeline,
        config: Optional[TrainingConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.pipeline = pipeline
        self.config = config or TrainingConfig()
        self.device = torch.device(device)
        self.loss_fn = FlowMatchingLoss(time_sampling=self.config.time_sampling)

    def train(self, dataset: LatentTextCacheDataset, output_dir: str | Path = "./checkpoints") -> List[float]:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        self.pipeline.core.train()
        self.pipeline.core.to(self.device)

        # Usar DataParallel si hay múltiples GPUs disponibles (ej. Kaggle T4 x2)
        n_gpus = torch.cuda.device_count() if self.device.type == "cuda" else 0
        net = nn.DataParallel(self.pipeline.core) if n_gpus > 1 else self.pipeline.core
        if n_gpus > 1:
            print(f"[FlowTrainer] Activando DataParallel en {n_gpus} GPUs.")

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        optimizer = torch.optim.AdamW(
            self.pipeline.core.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        scaler = torch.amp.GradScaler(self.device.type, enabled=(self.config.mixed_precision == "fp16"))
        losses = []

        print(f"[FlowTrainer] Iniciando entrenamiento del CORE por {self.config.epochs} épocas...")
        for epoch in range(self.config.epochs):
            total_loss = 0.0
            for step, (latents, txt_ctx, txt_mask) in enumerate(loader):
                latents = latents.to(self.device, dtype=torch.float32)
                txt_ctx = txt_ctx.to(self.device, dtype=torch.float32)
                txt_mask = txt_mask.to(self.device)

                with torch.amp.autocast(self.device.type, enabled=(self.config.mixed_precision == "fp16")):
                    loss = self.loss_fn(
                        model=net,
                        x0=latents,
                        txt_ctx=txt_ctx,
                        txt_mask=txt_mask,
                        cfg_dropout=self.config.cfg_dropout,
                    )

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.pipeline.core.parameters(), self.config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            losses.append(avg_loss)
            print(f"[Epoch {epoch + 1}/{self.config.epochs}] Loss: {avg_loss:.5f}")

            if (epoch + 1) % self.config.save_every_epochs == 0 or (epoch + 1) == self.config.epochs:
                save_dir = out_path / f"checkpoint-epoch-{epoch + 1}"
                self.pipeline.save_pretrained(save_dir)
                print(f"[FlowTrainer] Checkpoint guardado en: {save_dir}")

        return losses


class LoRATrainer:
    """
    Entrenador de adaptadores LoRA sobre el CORE.
    Congela el 98% de los parámetros y entrena únicamente matrices de bajo rango.
    """
    def __init__(
        self,
        pipeline: FlowCoderPipeline,
        lora_config: Optional[LoRAConfig] = None,
        train_config: Optional[TrainingConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.pipeline = pipeline
        self.lora_config = lora_config or LoRAConfig()
        self.train_config = train_config or TrainingConfig()
        self.device = torch.device(device)

        # Inyectar adaptadores y congelar base
        inject_lora(self.pipeline.core, self.lora_config)
        self.loss_fn = FlowMatchingLoss(time_sampling=self.train_config.time_sampling)

    def train(self, dataset: LatentTextCacheDataset, save_path: str | Path = "./lora_weights.safetensors") -> List[float]:
        self.pipeline.core.train()
        self.pipeline.core.to(self.device)

        n_gpus = torch.cuda.device_count() if self.device.type == "cuda" else 0
        net = nn.DataParallel(self.pipeline.core) if n_gpus > 1 else self.pipeline.core

        trainable_params = [p for p in self.pipeline.core.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.train_config.lr * 2.0)

        loader = DataLoader(dataset, batch_size=self.train_config.batch_size, shuffle=True)
        losses = []

        print(f"[LoRATrainer] Entrenando {sum(p.numel() for p in trainable_params)/1e6:.2f}M parámetros LoRA...")
        for epoch in range(self.train_config.epochs):
            total_loss = 0.0
            for latents, txt_ctx, txt_mask in loader:
                latents = latents.to(self.device, dtype=torch.float32)
                txt_ctx = txt_ctx.to(self.device, dtype=torch.float32)
                txt_mask = txt_mask.to(self.device)

                loss = self.loss_fn(
                    model=net,
                    x0=latents,
                    txt_ctx=txt_ctx,
                    txt_mask=txt_mask,
                    cfg_dropout=self.train_config.cfg_dropout,
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, self.train_config.max_grad_norm)
                optimizer.step()
                total_loss += loss.item()

            avg = total_loss / len(loader)
            losses.append(avg)
            print(f"[LoRA Epoch {epoch + 1}/{self.train_config.epochs}] Loss: {avg:.5f}")

        save_lora_weights(self.pipeline.core, save_path)
        print(f"[LoRATrainer] Pesos LoRA guardados exitosamente en: {save_path}")
        return losses


class EndToEndFinetuner:
    """
    Refinamiento Completo Extremo a Extremo (Joint Co-adaptation):
    Ajusta simultáneamente ENCODER + CORE + VAE con grupos de hiperparámetros y learning rates
    independientes. Permite acoplar de forma óptima modelos personalizados.
    """
    def __init__(
        self,
        pipeline: FlowCoderPipeline,
        config: Optional[EndToEndFinetuneConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.pipeline = pipeline
        self.config = config or EndToEndFinetuneConfig()
        self.device = torch.device(device)
        self.flow_loss_fn = FlowMatchingLoss()
        self.recon_loss_fn = EndToEndReconstructionLoss().to(self.device)

    def train_step(self, raw_images: torch.Tensor, texts: List[str], optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        raw_images = raw_images.to(self.device)

        # 1. Codificación de texto (gradientes activos para el encoder)
        txt_ctx, txt_mask = self.pipeline.text_encoder(texts, device=self.device)

        # 2. Codificación VAE de imagen real a latente limpio x0
        x0 = self.pipeline.vae.encode(raw_images)

        # 3. Flujo y muestreo temporal t
        B = x0.shape[0]
        t = torch.rand(B, device=self.device)
        t_expand = t[:, None, None, None]
        eps = torch.randn_like(x0)
        xt = (1.0 - t_expand) * x0 + t_expand * eps
        target_v = eps - x0

        # 4. Predicción del CORE
        pred_v = self.pipeline.core(xt, t, txt_ctx, txt_mask)
        flow_loss = nn.functional.mse_loss(pred_v, target_v)

        # 5. Reconstrucción perceptual extremo a extremo:
        # Estimar el latente limpio desde xt y la velocidad predicha: x0_hat = xt - t * pred_v
        x0_hat = xt - t_expand * pred_v
        recon_images = self.pipeline.vae.decode(x0_hat)
        recon_loss = self.recon_loss_fn(recon_images, raw_images)

        # 6. Pérdida conjunta ponderada
        total_loss = (
            self.config.flow_loss_weight * flow_loss +
            self.config.recon_loss_weight * recon_loss
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "flow_loss": flow_loss.item(),
            "recon_loss": recon_loss.item(),
        }

    def finetune(
        self,
        images_and_texts_loader: DataLoader,
        output_dir: str | Path = "./finetuned_pipeline",
    ) -> List[Dict[str, float]]:
        self.pipeline.to(self.device)
        self.pipeline.core.train()
        self.pipeline.vae.train()
        self.pipeline.text_encoder.train()

        # Configuración de optimizador con learning rates separados por componente
        param_groups = [
            {"params": self.pipeline.core.parameters(), "lr": self.config.lr_core},
        ]
        if not self.config.freeze_encoder:
            param_groups.append({"params": self.pipeline.text_encoder.parameters(), "lr": self.config.lr_encoder})
        else:
            for p in self.pipeline.text_encoder.parameters():
                p.requires_grad = False

        if not self.config.freeze_vae:
            param_groups.append({"params": self.pipeline.vae.parameters(), "lr": self.config.lr_vae})
        else:
            for p in self.pipeline.vae.parameters():
                p.requires_grad = False

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.config.weight_decay)

        history = []
        print(f"[EndToEndFinetuner] Iniciando ajuste extremo a extremo (LR Core: {self.config.lr_core}, "
              f"LR Encoder: {self.config.lr_encoder}, LR VAE: {self.config.lr_vae})...")

        for epoch in range(self.config.epochs):
            epoch_stats = {"total_loss": 0.0, "flow_loss": 0.0, "recon_loss": 0.0}
            for batch_imgs, batch_texts in images_and_texts_loader:
                stats = self.train_step(batch_imgs, batch_texts, optimizer)
                for k, v in stats.items():
                    epoch_stats[k] += v

            n_batches = len(images_and_texts_loader)
            for k in epoch_stats:
                epoch_stats[k] /= n_batches
            history.append(epoch_stats)

            print(f"[E2E Epoch {epoch + 1}/{self.config.epochs}] Total Loss: {epoch_stats['total_loss']:.4f} | "
                  f"Flow: {epoch_stats['flow_loss']:.4f} | Recon: {epoch_stats['recon_loss']:.4f}")

        self.pipeline.save_pretrained(output_dir)
        print(f"[EndToEndFinetuner] Pipeline completo afinado guardado en: {output_dir}")
        return history
