import unittest
import tempfile
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import torch.nn as nn

from flowcoder.config import CoreConfig, PipelineConfig, LoRAConfig, EndToEndFinetuneConfig
from flowcoder.models.core import MobileCoreDenoiser
from flowcoder.models.encoders import BaseTextEncoder, CustomTextEncoder
from flowcoder.models.vaes import TinyVAE, CustomVAE
from flowcoder.models.lora import inject_lora, merge_lora, get_lora_state_dict
from flowcoder.pipeline import FlowCoderPipeline
from flowcoder.solvers.flow_solver import FlowSolver
from flowcoder.training.losses import FlowMatchingLoss, EndToEndReconstructionLoss


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


if __name__ == "__main__":
    unittest.main()
