from typing import List, Optional, Tuple
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class LatentTextCacheDataset(Dataset):
    """Dataset que almacena latentes y embeddings precomputados en memoria o disco."""
    def __init__(self, latents: torch.Tensor, txt_embeddings: torch.Tensor, txt_masks: torch.Tensor):
        self.latents = latents              # (N, channels, H, W)
        self.txt_embeddings = txt_embeddings # (N, L, txt_dim)
        self.txt_masks = txt_masks          # (N, L)

    def __len__(self) -> int:
        return self.latents.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.latents[idx], self.txt_embeddings[idx], self.txt_masks[idx]


class CacheManager:
    """Precalcula y almacena en caché offline los embeddings de texto y latentes del VAE."""

    @staticmethod
    @torch.no_grad()
    def precompute(
        pipeline,
        images: torch.Tensor | List[Any],
        texts: List[str],
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> LatentTextCacheDataset:
        pipeline.to(device)
        pipeline.vae.eval()
        pipeline.text_encoder.eval()

        N = len(texts)
        lat_list = []
        emb_list = []
        mask_list = []

        print(f"[CacheManager] Precomputando latentes y embeddings para {N} muestras...")

        # 1. Precomputar texto
        for i in range(0, N, batch_size):
            batch_texts = texts[i:i + batch_size]
            emb, mask = pipeline.text_encoder(batch_texts, device=torch.device(device))
            emb_list.append(emb.cpu().half())
            mask_list.append(mask.cpu())

        # 2. Precomputar latentes de imágenes
        if isinstance(images, torch.Tensor):
            for i in range(0, N, batch_size):
                batch_imgs = images[i:i + batch_size].to(device)
                z = pipeline.vae.encode(batch_imgs)
                lat_list.append(z.cpu().half())
        else:
            # Lista de imágenes PIL o arrays
            from torchvision import transforms
            tx = transforms.Compose([
                transforms.Resize((pipeline.config.resolution, pipeline.config.resolution)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
            for i in range(0, N, batch_size):
                batch_imgs = torch.stack([tx(img) for img in images[i:i + batch_size]]).to(device)
                z = pipeline.vae.encode(batch_imgs)
                lat_list.append(z.cpu().half())

        all_latents = torch.cat(lat_list, dim=0)
        all_embs = torch.cat(emb_list, dim=0)
        all_masks = torch.cat(mask_list, dim=0)

        print(f"[CacheManager] Caché completada. Latentes: {all_latents.shape}, Embeddings: {all_embs.shape}")
        return LatentTextCacheDataset(all_latents, all_embs, all_masks)
