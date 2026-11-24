"""
Ejemplo 3: Finetuning Ligero con LoRA (Low-Rank Adaptation).
Permite adaptar el CORE a un nuevo estilo o concepto entrenando <2% de los pesos.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from flowcoder import FlowCoderPipeline, CoreConfig, LoRAConfig, TrainingConfig
from flowcoder.training import CacheManager, LoRATrainer

# 1. Cargar pipeline base
pipeline = FlowCoderPipeline.create(
    core_config=CoreConfig(dim=256, depth=6),
    vae="tiny",
)

# 2. Configurar LoRA sobre proyecciones de atención y capas pointwise
lora_cfg = LoRAConfig(
    r=16,
    alpha=32.0,
    dropout=0.05,
    target_modules=["to_q", "to_kv", "proj_out", "pw1", "pw2"],
)

# 3. Datos de ejemplo para adaptación de estilo
dummy_images = torch.randn(16, 3, 256, 256)
dummy_texts = ["anime portrait with colorful hair" for _ in range(16)]

dataset = CacheManager.precompute(pipeline, dummy_images, dummy_texts, batch_size=8)

# 4. Entrenar únicamente los adaptadores LoRA
trainer = LoRATrainer(pipeline=pipeline, lora_config=lora_cfg, train_config=TrainingConfig(epochs=5, batch_size=4))
trainer.train(dataset, save_path="./models/anime_style_lora.safetensors")

# 5. Aplicar y fusionar en inferencia
pipeline.load_lora("./models/anime_style_lora.safetensors")
pipeline.merge_lora_weights() # Fusión sin latencia extra
print("¡Adaptador LoRA entrenado y fusionado con éxito!")
