import math

import torch
import torch.nn as nn


def attention(q, k, v, causal=False):
    """
    q, v, v: [B, T, D]

    returns:
        output: [B, T, D]
    """
    d = q.shape[-1]

    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(d)

    if causal:
        T = q.shape[1]

        mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=q.device),
            diagonal=1,
        )

        scores = scores.masked_fill(mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)

    return weights @ v


def test_attention():
    B, T, D = 2, 4, 8

    q = torch.randn(B, T, D)
    k = torch.randn(B, T, D)
    v = torch.randn(B, T, D)

    y = attention(q, k, v, causal=True)

    print(y.shape)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        x: [B, T, D]
        """
        B, T, D = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [B, T, D]
        # ->
        # [B, T, H, D//H]
        # ->
        # [B, H, T, D//H]

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1)
        scores /= math.sqrt(self.head_dim)

        weights = torch.softmax(scores, dim=-1)

        out = weights @ v

        # [B, H, T, D//H]
        # ->
        # [B, T, H, D//H]
        # ->
        # [B, T, D]

        # transpose() changes the tensor strides without necessarily
        # rearranging the underlying memory. view() requires compatible
        # contiguous memory layout.
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, T, D)

        return self.out_proj(out)


def test_multi_head_attention():
    x = torch.randn(2, 5, 32)

    attn = MultiHeadAttention(d_model=32, n_heads=4)

    y = attn(x)

    print(y.shape)


if __name__ == "__main__":
    test_attention()

    test_multi_head_attention()
