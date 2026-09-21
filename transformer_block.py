import torch
import torch.nn as nn

from attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


def example_transformer_block():
    x = torch.randn(2, 10, 64)

    block = TransformerBlock(d_model=64, n_heads=8, d_ff=256)

    y = block(x)

    print(y.shape)


if __name__ == "__main__":
    example_transformer_block()
