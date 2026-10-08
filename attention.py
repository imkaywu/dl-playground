import math

import torch
import torch.nn as nn


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    q, v, v: [B, T, D]
    q:    [B, ..., T_q, D]
    k:    [B, ..., T_k, D]
    v:    [B, ..., T_k, D_v]
    mask: broadcastable to [B, ..., T_q, T_k]
          where True means "mask out this position".

    returns:
        output: [B, T_q, D_v]
    """
    # Similarity between each query and each key.
    # [B, ..., T_q, D] @ [B, ..., D, T_k] ->
    # [B, ..., T_q, T_k]
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(d)

    # Prevent attention to masked keys before softmax.
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    # Normalize over keys.
    weights = torch.softmax(scores, dim=-1)

    # Weighted sum of value vectors.
    # [B, ..., T_q, T_k] @ [B, ..., T_k, D_v] ->
    # [B, ..., T_q, D_v]
    return weights @ v


def test_scaled_dot_product_attention():
    B, T, D = 2, 4, 8

    q = torch.randn(B, T, D)
    k = torch.randn(B, T, D)
    v = torch.randn(B, T, D)

    # True above the diagonal means those future positions are masked.
    mask = torch.triu(
        torch.ones(T, T, dtype=torch.bool),
        diagonal=1,
    )

    y = scaled_dot_product_attention(q, k, v, mask=mask)

    assert y.shape == (B, T, D)


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads):
        super().__init__()

        assert model_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.model_dim = model_dim

        self.q_proj = nn.Linear(input_dim, model_dim)
        self.k_proj = nn.Linear(input_dim, model_dim)
        self.v_proj = nn.Linear(input_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, x, mask=None):
        """
        x:    [B, T, input_dim]
        mask: broadcastable to [B, H, T, T]

        return: [B, T, model_dim]
        """
        B, T, _ = x.shape
        D = self.model_dim
        H = self.num_heads
        Dh = self.head_dim

        # Project input into query, key, and value vectors
        q = self.q_proj(x)  # [B, T, D]
        k = self.k_proj(x)  # [B, T, D]
        v = self.v_proj(x)  # [B, T, D]

        # [B, T, D] -> [B, T, H, Dh] -> [B, H, T, Dh]
        q = q.view(B, T, H, Dh).transpose(1, 2)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)

        out = scaled_dot_product_attention(q, k, v, mask)

        # transpose() changes the tensor strides without necessarily
        # rearranging the underlying memory. view() requires compatible
        # contiguous memory layout.
        # [B, H, T, Dh] -> [B, T, H, Dh] -> [B, T, D]
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, T, D)

        return self.out_proj(out)


class MultiHeadAttention2(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads):
        super().__init__()

        assert (
            model_dim % num_heads == 0
        ), "Embedding dimension must be 0 modulo number of heads."

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        # One matrix multiplication produces all three projections
        self.qkv_proj = nn.Linear(input_dim, 3 * model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

        self._reset_params()

    def _reset_params(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.out_proj.weight)
        self.out_proj.bias.data.fill_(0)

    # Mask must be at least 2-dimensional with |seq_length x seq_length|.
    def forward(self, x, mask=None):
        """
        x:    [B, T, input_dim]
        mask: broadcastable to [B, H, T, T]
        """
        B, T, _ = x.size()

        # [B, T, input_dim] -> [B, T, 3*model_dim]
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        # [B, T, 3*H*Dh] -> [B, T, H, 3*Dh]
        qkv = qkv.reshape(B, T, self.num_heads, 3 * self.head_dim)

        # Move heads before sequence positions.
        # [B, T, H, 3*Dh] -> [B, H, T, 3*Dh]
        qkv = qkv.permute(0, 2, 1, 3)

        # Split the final dimension into Q, K, and V.
        # Each tensor: [B, H, T, Dh]
        q, k, v = qkv.chunk(3, dim=-1)

        # Compute attention independently for every head.
        # [B, H, T, Dh] -> [B, H, T, Dh]
        out = scaled_dot_product_attention(q, k, v, mask=mask)

        # Move sequence positions before heads, then merge heads.
        # [B, H, T, Dh] -> [B, T, H, Dh] -> [B, T, model_dim]
        out = out.transpose(1, 2).contiguous().reshape(B, T, self.model_dim)

        # Final output projection
        return self.out_proj(out)


def test_multi_head_attention():
    x = torch.randn(2, 5, 32)

    attn = MultiHeadAttention(input_dim=32, model_dim=64, num_heads=4)

    y = attn(x)

    assert y.shape == (2, 5, 64)

    attn_packed = MultiHeadAttention2(
        input_dim=32,
        model_dim=64,
        num_heads=4,
    )

    y_packed = attn_packed(x)
    print(y_packed.shape)

    assert y_packed.shape == (2, 5, 64)


if __name__ == "__main__":
    test_scaled_dot_product_attention()

    test_multi_head_attention()

    print("All attention tests passed.")
