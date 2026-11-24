# ⚡ FlowCoder: Framework Generativo Híbrido Ultra-Rápido

**FlowCoder** es una librería modular en Python para crear, entrenar, afinar y ejecutar modelos generativos de texto-a-imagen de **cómputo ultra-bajo** basados en **Rectified Flow (Flow Matching)** y una arquitectura **CORE híbrida (ConvNeXt Depthwise + Cross/Self-Attention)**.

Está diseñada específicamente para romper la necesidad de 50 pasos de difusión tradicionales y permitir la generación de imágenes de alta resolución ($512\times 512$, $1024\times 1024$) en **2 a 4 pasos (máximo 8)** con un consumo de recursos apto para CPUs y dispositivos móviles (Android / iOS / Edge).

---

## 🚀 Características Principales

1. **Inferencia en 2 a 4 pasos:**
   - Basado en **Rectified Flow (Optimal Transport)** con trayectorias de integración rectas ($\frac{dx}{dt} = \epsilon - x_0$).
   - Solvers numéricos integrados: **Euler (1-8 pasos)** y **Heun/Midpoint de 2.º orden (2-4 pasos)**.
   - **Guidance Distillation nativo:** El modelo admite la escala de guía como acondicionamiento, eliminando la duplicación del CFG a una sola pasada por paso.

2. **Arquitectura CORE Híbrida Móvil:**
   - **Patchify Stem ($2 \times 2$):** Reduce la resolución latente para que resoluciones como $512\times 512$ u $1024\times 1024$ no desborden la memoria.
   - **Convoluciones Depthwise $7 \times 7$ (ConvNeXt):** Aportan gran campo receptivo y coherencia espacial local con complejidad lineal $O(HW)$ y un ahorro del $85\%$ de operaciones frente a convoluciones estándar.
   - **Atención Dispersa / Asimétrica:** Cross-Attention regular con el texto y Self-Attention reservada a las capas más profundas mediante PyTorch SDPA / FlashAttention.
   - **Tiempo continuo $t \in [0, 1]$:** Modulación adaptativa (AdaLN) con embeddings sinusoidales.

3. **Modularidad Total (Pluggable Components):**
   - **Encoders:** Compatible con Hugging Face (`flan-t5`, `CLIP`, `MobileCLIP`, `BERT`) o modelos personalizados (`CustomTextEncoder`).
   - **VAEs:** Compatible con Diffusers (`sd-vae-ft-mse`), **TinyVAE / TAESD** (decodifica $512\times 512$ en <15 ms en CPU) o autoencoders propios (`CustomVAE`).
   - **CORE:** Altamente configurable en profundidad, dimensiones, cabezas, tamaños de parche y ratios de FFN.

4. **Tres Modos de Entrenamiento y Ajuste:**
   - **Preentrenamiento / Scratch:** `FlowTrainer` con caché offline de latentes y embeddings para entrenar hasta $50\times$ más rápido.
   - **LoRA (Low-Rank Adaptation):** `LoRATrainer` para adaptar estilos o dominios entrenando $<2\%$ de los parámetros con fusión sin coste en inferencia.
   - **Refinamiento Completo Extremo a Extremo:** `EndToEndFinetuner` con Learning Rates separados (`lr_encoder`, `lr_core`, `lr_vae`) y pérdida perceptual dual (Flow + Reconstrucción) para armonizar de forma natural componentes dispares.

---

## 📦 Estructura del Proyecto

```
IMAGEN/
├── flowcoder/
│   ├── __init__.py
│   ├── config.py              # Clases de configuración (CoreConfig, PipelineConfig, etc.)
│   ├── pipeline.py            # FlowCoderPipeline (Model Manager, inferencia, save/load)
│   ├── models/
│   │   ├── core.py            # MobileCoreDenoiser (ConvNeXt + Attention + AdaLN)
│   │   ├── encoders.py        # BaseTextEncoder, HFTextEncoder, CustomTextEncoder
│   │   ├── vaes.py            # BaseVAE, DiffusersVAE, TinyVAE, CustomVAE
│   │   └── lora.py            # Capas LoRA, inyección, merge y serialización
│   ├── solvers/
│   │   └── flow_solver.py     # Solvers Rectified Flow (Euler y Heun)
│   └── training/
│       ├── caching.py         # Caching offline de latentes y texto
│       ├── losses.py          # FlowMatchingLoss y EndToEndReconstructionLoss
│       └── trainer.py         # FlowTrainer, LoRATrainer, EndToEndFinetuner
├── examples/
│   ├── 01_inference_quickstart.py
│   ├── 02_train_custom_model.py
│   ├── 03_lora_finetuning.py
│   └── 04_end_to_end_harmonization.py
├── tests/
│   └── test_flowcoder.py      # Pruebas unitarias
└── setup.py
```

---

## 🛠️ Instalación

```bash
pip install -e .
```

---

## 💻 Guía Rápida de Uso

### 1. Inferencia Ultra-Rápida en 3 Pasos
```python
from flowcoder import FlowCoderPipeline, CoreConfig

# Cargar pipeline con TinyVAE para móvil o CPU
pipeline = FlowCoderPipeline.create(
    core_config=CoreConfig(dim=384, depth=12, patch_size=2),
    text_encoder="google/flan-t5-small",
    vae="tiny", # TinyVAE ultra-rápido
    resolution=512,
).to("cuda" if torch.cuda.is_available() else "cpu")

# Generar en 3 pasos usando Heun (2.º orden)
images = pipeline.generate(
    prompts="a cute electric yellow creature running in a forest",
    steps=3,
    solver="heun",
    cfg_scale=3.0,
)
images[0].save("output.png")
```

---

### 2. Entrenamiento del CORE con Caché Offline
```python
from flowcoder import FlowCoderPipeline, CoreConfig, TrainingConfig
from flowcoder.training import CacheManager, FlowTrainer

pipeline = FlowCoderPipeline.create(
    core_config=CoreConfig(dim=384, depth=8),
    vae="tiny",
    resolution=512,
)

# 1. Precomputar latentes y texto una sola vez
dataset = CacheManager.precompute(pipeline, images, captions, batch_size=16)

# 2. Entrenar el CORE a toda velocidad
trainer = FlowTrainer(pipeline, config=TrainingConfig(epochs=20, lr=2e-4))
trainer.train(dataset, output_dir="./checkpoints")
```

---

### 3. Adaptación Rápida con LoRA
```python
from flowcoder.config import LoRAConfig, TrainingConfig
from flowcoder.training import LoRATrainer

lora_cfg = LoRAConfig(r=16, alpha=32.0, target_modules=["to_q", "to_kv", "pw1", "pw2"])
trainer = LoRATrainer(pipeline, lora_config=lora_cfg, train_config=TrainingConfig(epochs=10))
trainer.train(dataset, save_path="./anime_lora.safetensors")

# Cargar y fusionar sin latencia extra en inferencia
pipeline.load_lora("./anime_lora.safetensors")
pipeline.merge_lora_weights()
```

---

### 4. Refinamiento Completo Extremo a Extremo (Armonización de Modelos)
Cuando integras un Encoder propio y un VAE propio que no fueron entrenados para esta tarea, puedes afinarlos conjuntamente:
```python
from flowcoder.config import EndToEndFinetuneConfig
from flowcoder.training import EndToEndFinetuner

e2e_cfg = EndToEndFinetuneConfig(
    lr_core=1e-4,       # LR medio para el CORE
    lr_encoder=1e-5,    # LR bajo para preservar conocimiento del lenguaje
    lr_vae=5e-6,        # LR bajo para ajuste sutil de textura
    epochs=5,
    flow_loss_weight=1.0,
    recon_loss_weight=0.5,
)

finetuner = EndToEndFinetuner(pipeline=pipeline, config=e2e_cfg)
finetuner.finetune(dataloader, output_dir="./harmonized_pipeline")
```
