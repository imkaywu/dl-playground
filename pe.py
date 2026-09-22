import torch


def apply_rope(x):
    """
    x: [B,H,T,D]

    D must be even.
    """

    B, H, T, D = x.shape

    half = D // 2

    x1 = x[..., :half]
    x2 = x[..., half:]

    positions = torch.arange(
        T,
        device=x.device,
        dtype=x.dtype,
    )

    dims = torch.arange(
        half,
        device=x.device,
        dtype=x.dtype,
    )

    theta = positions[:, None] / (10000 ** (2 * dims / D))

    cos = torch.cos(theta)
    sin = torch.sin(theta)

    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    rotated_1 = x1 * cos - x2 * sin

    rotated_2 = x1 * sin + x2 * cos

    return torch.cat(
        [rotated_1, rotated_2],
        dim=-1,
    )
