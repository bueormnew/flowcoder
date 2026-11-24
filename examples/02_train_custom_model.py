"""
Ejemplo 2: Entrenamiento del CORE desde cero con Caché de Embeddings.
Demuestra la velocidad 10x al precalcular latentes del VAE y texto de una vez.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from flowcoder import FlowCoderPipeline, CoreConfig, TrainingConfig
from flowcoder.training import CacheManager, FlowTrainer

# 1. Definir dimensiones y arquitectura del CORE propio
core_cfg = CoreConfig(
    dim=384,
    depth=8,
    patch_size=2,
    heads=6,
    dw_kernel_size=7,
)

# 2. Inicializar el pipeline con el VAE y Text Encoder deseados
pipeline = FlowCoderPipeline.create(
    core_config=core_cfg,
    text_encoder="google/flan-t5-small",
    vae="tiny",
    resolution=512,
)

# 3. Datos sintéticos de demostración (o carga tu propio dataset de imágenes y textos)
dummy_images = torch.randn(32, 3, 512, 512)
dummy_captions = [f"a photo of object {i} in high quality" for i in range(32)]

# 4. Precalcular caché (elimina el coste del VAE y T5 en las épocas de entrenamiento)
dataset = CacheManager.precompute(
    pipeline=pipeline,
    images=dummy_images,
    texts=dummy_captions,
    batch_size=8,
)

# 5. Entrenar el CORE con Rectified Flow Loss
train_cfg = TrainingConfig(
    epochs=10,
    batch_size=4,
    lr=2e-4,
    time_sampling="uniform",
    save_every_epochs=5,
)

trainer = FlowTrainer(pipeline=pipeline, config=train_cfg)
losses = trainer.train(dataset, output_dir="./models/my_custom_core")

print("Entrenamiento completado. Modelo guardado.")
