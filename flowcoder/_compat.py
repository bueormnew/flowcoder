from __future__ import annotations
import torch


def apply_torch_compat_patches() -> None:
    """
    Aplica parches de compatibilidad para entornos con versiones dispares de PyTorch y transformers
    (ej. Kaggle Python 3.13 + PyTorch) sin afectar el entrenamiento distribuido real cuando funciona.
    """
    # 1. Parche para torch._utils._chunk_or_narrow_cat (usado por versiones recientes de transformers)
    try:
        import torch._utils
        if not hasattr(torch._utils, "_chunk_or_narrow_cat"):
            def _chunk_or_narrow_cat(tensor, num_chunks, narrow_dim, cat_dim=0):
                return torch.cat(torch.chunk(tensor, num_chunks, dim=narrow_dim), dim=cat_dim)
            torch._utils._chunk_or_narrow_cat = _chunk_or_narrow_cat
    except Exception:
        pass

    # 2. Guardia para torch.distributed.tensor / _c10d_functional.wait_tensors
    # Sólo actúa si torch.distributed.tensor falla al importarse (ej. mismatch en Kaggle).
    dist_tensor_broken = False
    try:
        import torch.distributed.tensor  # noqa: F401
    except Exception:
        dist_tensor_broken = True

    if dist_tensor_broken:
        # Proveer fallback seguro para _c10d_functional.wait_tensors si falta
        try:
            c10d = getattr(torch.ops, "_c10d_functional", None)
            if c10d is not None and not hasattr(c10d, "wait_tensors"):
                c10d.wait_tensors = lambda tensor, *args, **kwargs: tensor
        except Exception:
            pass

        # Evitar que transformers intente cargar rutas de tensor_parallel rotas si no hay dist inicializado
        dist_initialized = False
        try:
            import torch.distributed as dist
            dist_initialized = dist.is_available() and dist.is_initialized()
        except Exception:
            dist_initialized = False

        if not dist_initialized:
            try:
                import transformers.utils.import_utils as import_utils
                import_utils.is_torch_distributed_available = lambda: False
            except Exception:
                pass
