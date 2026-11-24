import unittest
import tempfile
import sys
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import torch.nn as nn
from PIL import Image

from flowcoder._compat import apply_torch_compat_patches
from flowcoder.config import CoreConfig, PipelineConfig, TrainingConfig, LoRAConfig, EndToEndFinetuneConfig
from flowcoder.models.core import MobileCoreDenoiser
from flowcoder.models.encoders import BaseTextEncoder, CustomTextEncoder
from flowcoder.models.vaes import TinyVAE, CustomVAE
from flowcoder.models.lora import inject_lora, merge_lora, get_lora_state_dict, LoRALinear, LoRAConv2d
from flowcoder.pipeline import FlowCoderPipeline
from flowcoder.solvers.flow_solver import FlowSolver
from flowcoder.training.caching import CacheManager
from flowcoder.training.losses import FlowMatchingLoss, EndToEndReconstructionLoss
from flowcoder.training.trainer import FlowTrainer, LoRATrainer


class DummyTextEncoder(BaseTextEncoder):
    """Encoder de texto sintético para pruebas rápidas sin descargar pesos."""
    def __init__(self, dim=256, max_len=16):
        super().__init__()
        self._dim = dim
        self.max_len = max_len
        self.embedding = nn.Embedding(100, dim)

    @property
    def output_dim(self) -> int:
        return self._dim

    def forward(self, texts, device):
        B = len(texts) if isinstance(texts, list) else texts.shape[0]
        # Generar tokens falsos
        tokens = torch.randint(0, 100, (B, self.max_len), device=device)
        mask = torch.ones((B, self.max_len), dtype=torch.bool, device=device)
        return self.embedding(tokens), mask


class TestFlowCoder(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.core_cfg = CoreConfig(
            in_channels=4,
            out_channels=4,
            patch_size=2,
            dim=128,              # Reducido para pruebas ultra-rápidas
            depth=4,
            heads=4,
            txt_dim=256,
            cross_attn_interval=1,
            self_attn_layers=[2, 3],
            dw_kernel_size=7,
        )

    def test_core_forward(self):
        core = MobileCoreDenoiser(self.core_cfg)
        x = torch.randn(2, 4, 32, 32)
        t = torch.tensor([0.2, 0.8])
        txt = torch.randn(2, 16, 256)
        mask = torch.ones(2, 16, dtype=torch.bool)
        guidance = torch.tensor([3.0, 3.0])

        out = core(x, t, txt, mask, guidance=guidance)
        self.assertEqual(out.shape, (2, 4, 32, 32))

    def test_pipeline_generate(self):
        encoder = DummyTextEncoder(dim=256)
        vae = TinyVAE(latent_channels=4, base_channels=32)
        core = MobileCoreDenoiser(self.core_cfg)

        pipe = FlowCoderPipeline(core=core, text_encoder=encoder, vae=vae)
        pipe.to("cpu")

        # Probar generación de 2 pasos con solver Euler
        images = pipe.generate(["a blue water creature"], steps=2, solver="euler", height=128, width=128)
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].size, (128, 128))

        # Probar generación con solver Heun
        images_heun = pipe.generate(["a fiery red dragon"], steps=2, solver="heun", height=128, width=128)
        self.assertEqual(len(images_heun), 1)
        self.assertEqual(images_heun[0].size, (128, 128))

    def test_lora_injection_and_merge(self):
        core = MobileCoreDenoiser(self.core_cfg)
        lora_cfg = LoRAConfig(r=8, alpha=16.0, target_modules=["to_q", "to_kv", "pw1", "pw2"])

        # Contar parámetros antes de LoRA
        total_params = sum(p.numel() for p in core.parameters())

        inject_lora(core, lora_cfg)
        trainable_params = sum(p.numel() for p in core.parameters() if p.requires_grad)

        self.assertGreater(trainable_params, 0)
        self.assertLess(trainable_params, total_params * 0.1)  # Menos del 10% entrenable

        # Probar extracción de pesos LoRA
        state = get_lora_state_dict(core)
        self.assertTrue(any("lora_" in k for k in state.keys()))

        # Probar merge
        merge_lora(core)

    def test_losses_and_backward(self):
        core = MobileCoreDenoiser(self.core_cfg)
        loss_fn = FlowMatchingLoss()

        x0 = torch.randn(2, 4, 32, 32, requires_grad=True)
        txt = torch.randn(2, 16, 256)
        mask = torch.ones(2, 16, dtype=torch.bool)

        loss = loss_fn(core, x0, txt, mask)
        loss.backward()
        self.assertFalse(torch.isnan(loss))

    def test_save_and_load_pretrained(self):
        encoder = DummyTextEncoder(dim=256)
        vae = TinyVAE(latent_channels=4, base_channels=32)
        core = MobileCoreDenoiser(self.core_cfg)
        pipe = FlowCoderPipeline(core=core, text_encoder=encoder, vae=vae)

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            self.assertTrue((Path(tmpdir) / "pipeline_config.json").exists())

    def test_end_to_end_finetuner_step(self):
        from flowcoder.training.trainer import EndToEndFinetuner
        encoder = DummyTextEncoder(dim=256)
        vae = TinyVAE(latent_channels=4, base_channels=32)
        core = MobileCoreDenoiser(self.core_cfg)
        pipe = FlowCoderPipeline(core=core, text_encoder=encoder, vae=vae)

        finetuner = EndToEndFinetuner(pipe)
        optimizer = torch.optim.AdamW([
            {"params": pipe.core.parameters(), "lr": 1e-4},
            {"params": pipe.text_encoder.parameters(), "lr": 1e-5},
            {"params": pipe.vae.parameters(), "lr": 5e-6},
        ])

        dummy_imgs = torch.randn(2, 3, 64, 64)
        dummy_texts = ["test prompt one", "test prompt two"]
        stats = finetuner.train_step(dummy_imgs, dummy_texts, optimizer)

        self.assertIn("total_loss", stats)
        self.assertIn("flow_loss", stats)
        self.assertIn("recon_loss", stats)
        self.assertFalse(torch.isnan(torch.tensor(stats["total_loss"])))

    def test_compat_patches_chunk_and_broken_dist(self):
        # 1. Verificar _chunk_or_narrow_cat
        orig_fn = getattr(torch._utils, "_chunk_or_narrow_cat", None)
        try:
            if hasattr(torch._utils, "_chunk_or_narrow_cat"):
                delattr(torch._utils, "_chunk_or_narrow_cat")
            apply_torch_compat_patches()
            self.assertTrue(hasattr(torch._utils, "_chunk_or_narrow_cat"))
            t = torch.randn(2, 4)
            out = torch._utils._chunk_or_narrow_cat(t, 2, narrow_dim=1, cat_dim=0)
            self.assertEqual(out.shape, (4, 2))
        finally:
            if orig_fn is not None:
                torch._utils._chunk_or_narrow_cat = orig_fn

        # 2. Simular fallo de importación de torch.distributed.tensor por wait_tensors
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch.distributed.tensor" or name.startswith("torch.distributed.tensor."):
                raise AttributeError("'_OpNamespace' '_c10d_functional' object has no attribute 'wait_tensors'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=fake_import):
            apply_torch_compat_patches()
            c10d = getattr(torch.ops, "_c10d_functional", None)
            if c10d is not None:
                self.assertTrue(hasattr(c10d, "wait_tensors"))

    def test_core_bfloat16_and_int_mask(self):
        core = MobileCoreDenoiser(self.core_cfg).to(dtype=torch.bfloat16)
        x = torch.randn(2, 4, 32, 32, dtype=torch.bfloat16)
        t = torch.tensor([0.2, 0.8])
        txt = torch.randn(2, 16, 256, dtype=torch.bfloat16)
        mask = torch.ones(2, 16, dtype=torch.long)  # Máscara entera (ej. tokenizer)
        out = core(x, t, txt, mask, guidance=3.0)
        self.assertEqual(out.shape, (2, 4, 32, 32))
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_lora_conv2d_and_merge_idempotent(self):
        core = MobileCoreDenoiser(self.core_cfg)
        core.eval()
        lora_cfg = LoRAConfig(r=4, alpha=8.0, target_modules=["stem", "to_q"])
        inject_lora(core, lora_cfg)

        has_conv_lora = any(isinstance(m, LoRAConv2d) for m in core.modules())
        has_lin_lora = any(isinstance(m, LoRALinear) for m in core.modules())
        self.assertTrue(has_conv_lora)
        self.assertTrue(has_lin_lora)

        # Asignar pesos no nulos a lora_B para comprobar que merge no duplica el delta
        for m in core.modules():
            if isinstance(m, (LoRALinear, LoRAConv2d)):
                nn.init.normal_(m.lora_B, std=0.01)

        x = torch.randn(1, 4, 32, 32)
        t = torch.tensor([0.5])
        txt = torch.randn(1, 16, 256)
        mask = torch.ones(1, 16, dtype=torch.bool)

        with torch.no_grad():
            out_before = core(x, t, txt, mask)
            merge_lora(core)
            out_after = core(x, t, txt, mask)
            merge_lora(core)
            out_after_second = core(x, t, txt, mask)

        self.assertTrue(torch.allclose(out_before, out_after, atol=1e-5))
        self.assertTrue(torch.allclose(out_after, out_after_second, atol=1e-5))

    def test_pipeline_create_config_and_roundtrip(self):
        encoder = DummyTextEncoder(dim=256)
        vae = TinyVAE(latent_channels=4, base_channels=32)
        pipe = FlowCoderPipeline.create(
            core_config=self.core_cfg,
            text_encoder=encoder,
            vae=vae,
            resolution=128,
        )
        self.assertEqual(pipe.config.encoder_type, "custom")
        self.assertEqual(pipe.config.vae_type, "taesd")
        self.assertEqual(pipe.config.vae_name_or_path, "tiny")
        self.assertEqual(pipe.config.resolution, 128)

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            loaded_cfg = PipelineConfig.from_json(Path(tmpdir) / "pipeline_config.json")
            self.assertEqual(loaded_cfg.vae_name_or_path, "tiny")

            # Sin text_encoder explícito debe fallar claramente para encoder='custom'
            with self.assertRaises(ValueError):
                FlowCoderPipeline.from_pretrained(tmpdir)

            loaded = FlowCoderPipeline.from_pretrained(tmpdir, text_encoder=encoder)
            self.assertEqual(loaded.config.resolution, 128)

    def test_flow_solver_wrapped_module_and_validation(self):
        core = MobileCoreDenoiser(self.core_cfg)

        class DPWrapper(nn.Module):
            def __init__(self, mod):
                super().__init__()
                self.module = mod

            def forward(self, *args, **kwargs):
                return self.module(*args, **kwargs)

        wrapped = DPWrapper(core)
        latents = torch.randn(1, 4, 16, 16)
        txt = torch.randn(1, 16, 256)
        mask = torch.ones(1, 16, dtype=torch.bool)

        out_e = FlowSolver.sample_euler(wrapped, latents, txt, mask, steps=2, cfg_scale=2.0, one_pass_guidance=False)
        out_h = FlowSolver.sample_heun(wrapped, latents, txt, mask, steps=2, cfg_scale=2.0, one_pass_guidance=False)
        self.assertEqual(out_e.shape, latents.shape)
        self.assertEqual(out_h.shape, latents.shape)

        with self.assertRaises(ValueError):
            FlowSolver.sample_euler(wrapped, latents, txt, mask, steps=0)

    def test_caching_and_trainers_small_dataset_cpu(self):
        encoder = DummyTextEncoder(dim=256)
        vae = TinyVAE(latent_channels=4, base_channels=32)
        pipe = FlowCoderPipeline.create(
            core_config=self.core_cfg,
            text_encoder=encoder,
            vae=vae,
            resolution=64,
        )

        # Imágenes RGBA para verificar conversión a RGB
        imgs = [Image.new("RGBA", (64, 64), color=(10, 20, 30, 255)) for _ in range(3)]
        texts = ["one", "two", "three"]
        ds = CacheManager.precompute(pipe, imgs, texts, batch_size=2, device="cpu")
        self.assertEqual(len(ds), 3)

        with tempfile.TemporaryDirectory() as tmpdir:
            # batch_size > len(ds) y mixed_precision='fp16' en CPU no deben fallar
            t_cfg = TrainingConfig(epochs=1, batch_size=8, mixed_precision="fp16", save_every_epochs=1)
            trainer = FlowTrainer(pipe, config=t_cfg, device="cpu")
            losses = trainer.train(ds, output_dir=tmpdir)
            self.assertEqual(len(losses), 1)

            lora_path = Path(tmpdir) / "lora.safetensors"
            l_trainer = LoRATrainer(
                pipe,
                lora_config=LoRAConfig(r=4, alpha=8.0, target_modules=["to_q"]),
                train_config=TrainingConfig(epochs=1, batch_size=8),
                device="cpu",
            )
            l_losses = l_trainer.train(ds, save_path=lora_path)
            self.assertEqual(len(l_losses), 1)
            self.assertTrue(lora_path.exists())

    def test_custom_encoder_tuple_texts(self):
        emb = nn.Embedding(50, 256)

        def tok_fn(texts, max_length, device):
            tokens = torch.zeros((len(texts), max_length), dtype=torch.long, device=device)
            mask = torch.ones((len(texts), max_length), dtype=torch.bool, device=device)
            return tokens, mask

        enc = CustomTextEncoder(emb, output_dim=256, tokenizer_fn=tok_fn, max_length=8)
        out, mask = enc(("hello", "world"), torch.device("cpu"))
        self.assertEqual(out.shape, (2, 8, 256))
        self.assertEqual(mask.shape, (2, 8))


if __name__ == "__main__":
    unittest.main()
