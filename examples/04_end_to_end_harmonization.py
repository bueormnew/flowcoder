"""
Ejemplo 4: Refinamiento Extremo a Extremo (Joint Co-adaptation).
Ajusta simultáneamente un Text Encoder Propio, un CORE Propio y un VAE Propio
con Learning Rates separados para armonizarlos de manera natural hacia la tarea.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from flowcoder import FlowCoderPipeline, CoreConfig, EndToEndFinetuneConfig
from flowcoder.models.encoders import CustomTextEncoder
from flowcoder.models.vaes import TinyVAE
from flowcoder.training import EndToEndFinetuner


# 1. Definir un Text Encoder personalizado propio
class MyCustomLanguageModel(nn.Module):
    def __init__(self, vocab_size=5000, dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=4, batch_first=True),
            num_layers=2
        )

    def forward(self, token_ids):
        return self.transformer(self.embedding(token_ids))

custom_encoder_module = MyCustomLanguageModel()
custom_encoder = CustomTextEncoder(custom_encoder_module, output_dim=256)

# 2. VAE ultra-ligero y CORE compatible con las dimensiones del encoder
custom_vae = TinyVAE(latent_channels=4, base_channels=32)
custom_core_config = CoreConfig(dim=256, depth=6, txt_dim=256, patch_size=2)

pipeline = FlowCoderPipeline.create(
    core_config=custom_core_config,
    text_encoder=custom_encoder,
    vae=custom_vae,
    resolution=256,
)

# 3. Dataset conjunto de imágenes reales y tokens/textos
class DummyImageTextDataset(Dataset):
    def __len__(self):
        return 20

    def __getitem__(self, idx):
        img = torch.randn(3, 256, 256)
        token_ids = torch.randint(0, 5000, (16,))
        return img, token_ids

loader = DataLoader(DummyImageTextDataset(), batch_size=4, shuffle=True)

# 4. Configurar el afinamiento conjunto con LRs diferenciales
e2e_cfg = EndToEndFinetuneConfig(
    lr_core=1e-4,       # LR medio para el CORE
    lr_encoder=1e-5,    # LR bajo para no desestabilizar la semántica del lenguaje
    lr_vae=5e-6,        # LR muy bajo para afinar sutilmente la textura visual
    epochs=3,
    flow_loss_weight=1.0,
    recon_loss_weight=0.5,
)

finetuner = EndToEndFinetuner(pipeline=pipeline, config=e2e_cfg)
history = finetuner.finetune(loader, output_dir="./models/harmonized_pipeline")

print("¡Afinamiento conjunto completado! Los 3 modelos han sido armonizados.")
