import torch


def top_k_sample(logits, k, temperature=1.0):
    """
    logits: [V]
    """
    logits = logits / temperature

    values, indices = torch.topk(logits, k)

    probs = torch.softmax(values, dim=-1)

    sample_idx = torch.multinomial(probs, num_samples=1)

    return indices[sample_idx]


def test_top_k_sample():
    logits = torch.randn(1000)

    token = top_k_sample(logits, k=50, temperature=0.8)

    print(token)


def top_p_sample(logits, p=0.9, temperature=1.0):
    logits = logits / temperature

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)

    probs = torch.softmax(sorted_logits, dim=-1)

    cumulative_probs = torch.cumsum(probs, dim=-1)

    remove = cumulative_probs > p

    # We generally want to retain the token that crosses the threshold,
    # rather than stopping before it.
    remove[1:] = remove[:-1].clone()
    # Always keep at least the highest-probability token.
    remove[0] = False

    sorted_logits[remove] = float("-inf")

    probs = torch.softmax(sorted_logits, dim=-1)

    sample_idx = torch.multinomial(probs, num_samples=1)

    return sorted_indices[sample_idx]


def test_top_p_sample():
    logits = torch.randn(1000)

    token = top_p_sample(logits, p=0.8, temperature=0.8)

    print(token)


if __name__ == "__main__":
    test_top_k_sample()

    test_top_p_sample()
