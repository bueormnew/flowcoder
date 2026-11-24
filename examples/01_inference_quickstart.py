"""
Ejemplo 1: Inferencia Ultra-Rápida en 2 a 4 pasos con FlowCoder.
Demuestra cómo cargar un pipeline y generar imágenes de alta resolución.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flowcoder import FlowCoderPipeline, CoreConfig

# 1. Crear o cargar el pipeline
# Usa TinyVAE para latencia ultrabaja en CPU/móvil o 'stabilityai/sd-vae-ft-mse'
pipe = FlowCoderPipeline.create(
    core_config=CoreConfig(dim=384, depth=12, patch_size=2),
    text_encoder="google/flan-t5-small",
    vae="tiny", # TinyVAE: decodifica 512x512 en <15ms
    resolution=512,
)

# 2. Configurar acelerador (CPU, CUDA, MPS para Mac/iOS)
device = "cuda" if pipe.core.parameters().__next__().is_cuda else "cpu"
pipe.to(device)

print(f"Generando imagen en dispositivo: {device}...")

# 3. Generar en sólo 3 pasos con el solver Heun de 2.º orden
images = pipe.generate(
    prompts="a vibrant futuristic sports car, cyberpunk neon lights, 8k render",
    steps=3,               # Solo 3 pasos
    cfg_scale=3.0,         # Classifier-free guidance
    solver="heun",         # Solver predictor-corrector
    seed=42,
    output_type="pil",
)

images[0].save("output_3steps.png")
print("¡Imagen generada exitosamente y guardada en 'output_3steps.png'!")
