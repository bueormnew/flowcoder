import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowMatchingLoss(nn.Module):
    """
    Función de pérdida para Rectified Flow (Optimal Transport Flow Matching).
    Alinea el vector de velocidad v_pred con v_target = eps - x0 sobre la geodésica recta.
    """
    def __init__(self, time_sampling: str = "uniform"):
        super().__init__()
        self.time_sampling = time_sampling

    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.time_sampling == "logit_normal":
            u = torch.randn(batch_size, device=device)
            return torch.sigmoid(u)
        elif self.time_sampling == "beta":
            # Favorece tiempos intermedios
            dist = torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.5))
            return dist.sample((batch_size,)).to(device)
        else:
            # Distribución uniforme estándar U(0, 1)
            return torch.rand(batch_size, device=device)

    def forward(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        txt_ctx: torch.Tensor,
        txt_mask: torch.Tensor,
        cfg_dropout: float = 0.1,
        guidance: torch.Tensor = None,
    ) -> torch.Tensor:
        B = x0.shape[0]
        device = x0.device

        # 1. Muestreo de tiempo continuo t in [0, 1]
        t = self.sample_time(B, device)
        t_expand = t[:, None, None, None]

        # 2. Ruido gaussiano
        eps = torch.randn_like(x0)

        # 3. Interpolación recta: xt = (1 - t) * x0 + t * eps
        xt = (1.0 - t_expand) * x0 + t_expand * eps

        # 4. Velocidad objetivo recta: v = eps - x0
        target_v = eps - x0

        # 5. Classifier-Free Guidance Dropout auténtico
        if cfg_dropout > 0.0:
            drop = torch.rand(B, device=device) < cfg_dropout
            if drop.any():
                null_tokens = model.null_text_token.expand(drop.sum(), txt_ctx.shape[1], -1)
                txt_ctx = txt_ctx.clone()
                txt_ctx[drop] = null_tokens

        # 6. Predicción de velocidad del CORE
        pred_v = model(xt, t, txt_ctx, txt_mask, guidance=guidance)

        # 7. Pérdida MSE
        return F.mse_loss(pred_v, target_v)


class EndToEndReconstructionLoss(nn.Module):
    """
    Pérdida de reconstrucción compuesta (L1 + Gradiente Espacial / Sobel)
    para el ajuste extremo a extremo de VAE + CORE + TextEncoder.
    """
    def __init__(self):
        super().__init__()
        # Filtros Sobel para preservar bordes y texturas de alta frecuencia sin depender de VGG
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3) / 4.0
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3) / 4.0
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Extrae gradientes espaciales canal por canal
        B, C, H, W = pred.shape
        loss = 0.0
        for c in range(C):
            p = pred[:, c:c+1]
            t = target[:, c:c+1]
            gx_p = F.conv2d(p, self.sobel_x, padding=1)
            gy_p = F.conv2d(p, self.sobel_y, padding=1)
            gx_t = F.conv2d(t, self.sobel_x, padding=1)
            gy_t = F.conv2d(t, self.sobel_y, padding=1)
            loss += F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)
        return loss / C

    def forward(self, pred_imgs: torch.Tensor, target_imgs: torch.Tensor) -> torch.Tensor:
        l1 = F.l1_loss(pred_imgs, target_imgs)
        grad = self._gradient_loss(pred_imgs, target_imgs)
        return l1 + 0.5 * grad
