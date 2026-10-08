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
        x = x + self.attn(self.norm1(x))

        # Pre-LN feed-forward + residual connection.
        x = x + self.ffn(self.norm2(x))

        return x


def example_encoder_block():
    x = torch.randn(2, 10, 64)

    block = EncoderBlock(model_dim=64, num_heads=8, ff_dim=256)

    y = block(x)

    print(y.shape)
    print("Encoder block test passed")


if __name__ == "__main__":
    example_encoder_block()
