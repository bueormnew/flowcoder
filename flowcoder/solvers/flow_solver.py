from typing import Optional, Callable
import torch
import torch.nn as nn


class FlowSolver:
    """
    Solvers de integración numérica para Rectified Flow (Euler y Heun/Midpoint).
    Diseñados para generar imágenes de máxima nitidez en 1, 2, 3, 4 u 8 pasos.
    """

    @staticmethod
    @torch.no_grad()
    def sample_euler(
        model: nn.Module,
        latents: torch.Tensor,
        txt_ctx: torch.Tensor,
        txt_mask: Optional[torch.Tensor] = None,
        steps: int = 4,
        cfg_scale: float = 3.0,
        one_pass_guidance: bool = True,
        callback: Optional[Callable[[int, torch.Tensor], None]] = None,
    ) -> torch.Tensor:
        """
        Integrador Euler de 1.er orden: x_{t - dt} = x_t - v * dt.
        Ideal para 1 a 4 pasos ultra-rápidos.
        """
        dt = 1.0 / steps
        x = latents

        for step in range(steps):
            t_val = 1.0 - (step * dt)

            if one_pass_guidance and hasattr(model, "guidance_embed") and model.guidance_embed is not None:
                # 1-pass guidance: Se evalúa el modelo una sola vez por paso
                v = model(x, t_val, txt_ctx, txt_mask, guidance=cfg_scale)
            else:
                # 2-pass CFG tradicional
                if cfg_scale > 1.0:
                    v_cond = model(x, t_val, txt_ctx, txt_mask)
                    null_txt = model.null_text_token.expand(txt_ctx.shape[0], txt_ctx.shape[1], -1)
                    v_uncond = model(x, t_val, null_txt, None)
                    v = v_uncond + cfg_scale * (v_cond - v_uncond)
                else:
                    v = model(x, t_val, txt_ctx, txt_mask)

            # Paso de Euler hacia t = 0
            x = x - v * dt

            if callback is not None:
                callback(step, x)

        return x

    @staticmethod
    @torch.no_grad()
    def sample_heun(
        model: nn.Module,
        latents: torch.Tensor,
        txt_ctx: torch.Tensor,
        txt_mask: Optional[torch.Tensor] = None,
        steps: int = 2,
        cfg_scale: float = 3.0,
        one_pass_guidance: bool = True,
    ) -> torch.Tensor:
        """
        Integrador Heun de 2.° orden (Predictor-Corrector).
        Excelente fidelidad en 2 a 4 pasos gracias a corrección de curvatura residual.
        """
        dt = 1.0 / steps
        x = latents

        def get_v(curr_x, curr_t):
            if one_pass_guidance and hasattr(model, "guidance_embed") and model.guidance_embed is not None:
                return model(curr_x, curr_t, txt_ctx, txt_mask, guidance=cfg_scale)
            if cfg_scale > 1.0:
                v_c = model(curr_x, curr_t, txt_ctx, txt_mask)
                null_txt = model.null_text_token.expand(txt_ctx.shape[0], txt_ctx.shape[1], -1)
                v_u = model(curr_x, curr_t, null_txt, None)
                return v_u + cfg_scale * (v_c - v_u)
            return model(curr_x, curr_t, txt_ctx, txt_mask)

        for step in range(steps):
            t_curr = 1.0 - (step * dt)
            t_next = max(0.0, t_curr - dt)

            # Predictor (Euler)
            v1 = get_v(x, t_curr)
            x_pred = x - v1 * dt

            if t_next > 0:
                # Corrector (Midpoint/Heun)
                v2 = get_v(x_pred, t_next)
                x = x - 0.5 * (v1 + v2) * dt
            else:
                x = x_pred

        return x
