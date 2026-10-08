import torch
import torch.nn as nn

from attention import MultiHeadAttention
from ffn import FeedForwardNetwork
from norm import LayerNorm


class EncoderBlock(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim):
        super().__init__()

        # Normalize the iput before each layer
        self.norm1 = LayerNorm(model_dim)
        self.norm2 = LayerNorm(model_dim)

        # Sublayer 1: multi-head self-attention
        self.attn = MultiHeadAttention(
            input_dim=model_dim, model_dim=model_dim, num_heads=num_heads
        )

        # Sublayer 2: position-wise feed-forward network.
        self.ffn = FeedForwardNetwork(model_dim, ff_dim)

    def forward(self, x, mask=None):
        # Pre-LN self-attention + residual connection.
        x = x + self.attn(self.norm1(x), mask=mask)

        # Pre-LN feed-forward + residual connection.
        x = x + self.ffn(self.norm2(x))

        return x


def test_encoder_block():
    x = torch.randn(2, 10, 64)

    block = EncoderBlock(model_dim=64, num_heads=8, ff_dim=256)

    y = block(x)

    print(y.shape)
    print("Encoder block test passed")


class Encoder(nn.Module):
    def __init__(
        self,
        model_dim,
        num_heads,
        ff_dim,
        num_layers,
    ):
        super().__init__()

        # Create num_layers independent EncoderBlock instances.
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    model_dim=model_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask=None):
        """
        x: [B, T, D]
        mask: optional attention mask

        return [B, T, D]
        """

        # Pass the output of each block into the next block
        for layer in self.layers:
            x = layer(x, mask=mask)

        return x


def test_encoder():
    B, T, D = 2, 10, 64

    x = torch.randn(B, T, D)

    encoder = Encoder(
        model_dim=D,
        num_heads=4,
        ff_dim=256,
        num_layers=3,
    )

    y = encoder(x)

    assert y.shape == (B, T, D)
    print("Encoder test passed:", y.shape)


if __name__ == "__main__":
    test_encoder_block()

    test_encoder()
