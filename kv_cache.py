import torch

"""
When generating sequence: A, B, C. With KV caching:

step 1:
K = [K_A]
V = [V_A]

step 2:
K = [K_A, K_B]
V = [V_A, V_B]

step 3:
K = [K_A, K_B, K_C]
V = [V_A, V_B, V_C]
"""


def attention_with_kv_cache(
    q,
    k_new,
    v_new,
    cache_k=None,
    cache_v=None,
):
    """
    q:
        [B,H,1,D]

    k_new/v_new:
        [B,H,1,D]

    k/v:
        [B,H,T,D]
    """

    if cache_k is None:
        k = k_new
        v = v_new
    else:
        k = torch.cat(
            [cache_k, k_new],
            dim=2,
        )

        v = torch.cat(
            [cache_v, v_new],
            dim=2,
        )

    # [B,H,1,D] x [B,H,D,T] = [B,H,1,T]
    scores = q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5)

    weights = torch.softmax(
        scores,
        dim=-1,
    )

    output = weights @ v

    return output, k, v
