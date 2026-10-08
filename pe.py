import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super().__init__()

        # Store one positional vector for each possible position.
        # Shape: [max_len, model_dim]
        pe = torch.zeros(max_len, model_dim)

        for pos in range(max_len):
            for i in range(model_dim // 2):
                # Calculate the frequency for this dimension pair.
                # omega_i = 1 / 10000^(2i / D)
                omega = 1.0 / (10000 ** (2 * i / model_dim))

                # Calculate the angle for this position and frequency.
                # angle = pos * omega
                angle = pos * omega

                # Even dimension: PE[pos, 2i] = sin(angle)
                pe[pos, 2 * i] = math.sin(angle)

                # Odd dimension: PE[pos, 2i+1] = cos(angle)
                pe[pos, 2 * i + 1] = math.cos(angle)

        # Add a batch dimension so it broadcasts over [B, T, D].
        # [max_len, D] -> [1, max_len, D]
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """
        x: [B, T, D]
        returns: [B, T, D]
        """
        T = x.shape[1]

        # Add the first T positional vectors to the token embeddings.
        # [B, T, D] + [1, T, D] -> [B, T, D]
        return x + self.pe[:, :T, :]


def test_positional_encoding():
    x = torch.zeros(2, 4, 4)

    pe = SinusoidalPositionalEncoding(
        model_dim=4,
        max_len=10,
    )

    y = pe(x)

    print(y[0])  # Display the positional vectors for the first sample.
    assert y.shape == x.shape


# TODO: add visualization
class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super().__init__()

        # Position indices: [max_len, 1]
        # Each row corresponds to one sequence position.
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)

        # Frequency for each pair of dimensions
        #
        # 1 / 10000^(2i / D) -> 10000^(-2i/D)
        # ->
        # e^[(-2i/D)*log(10000)]
        #
        # i = 0, 1, ..., floor(D/2)-1
        #
        # Shape: ceil(D/2) for odd D, floor(D/2) for even D.
        div_term = torch.exp(
            torch.arange(0, model_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / model_dim)
        )

        # Alternative implementation
        # div_term = 1.0 / (
        #     10000.0
        #     ** (torch.arange(0, model_dim, 2, dtype=torch.float32) /
        #     model_dim)
        # )

        # Store the positional encoding as [max_len, D]
        pe = torch.zeros(max_len, model_dim)

        # Even dimensions: PE[pos, 2i] = sin(pos / 10000^(2i/D))
        # Shape: ceil(D/2)
        pe[:, 0::2] = torch.sin(position * div_term)

        # Odd dimensions: PE[pos, 2i+1] = cos(pos / 10000^(2i/D))
        # Shape: floor(D/2)
        pe[:, 1::2] = torch.cos(position * div_term[: model_dim // 2])

        # Register as a buffer: saved with the model, but nor trainable.
        # Shape: [1, max_len, D] for broadcasting over the batch
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """
        x: [B, T, D]
        returns: [B, T, D]
        """

        # Select positional encodings for position 0 through T-1.
        # [1, T, D] broadcasts across the bach dimension B.
        T = x.shape[1]

        return x + self.pe[:, :T, :]


def test_sinusoidal_positional_encoding():
    B, T, D = 2, 5, 32

    x = torch.zeros(B, T, D)
    pe = SinusoidalPositionalEncoding(model_dim=D)

    y = pe(x)

    assert y.shape == x.shape

    assert torch.allclose(y[0], y[1])

    print("Sinusoidal positional encoding test passed")


# TODO: rewrite as a class
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
