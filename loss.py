import torch
import torch.nn.functional as F


def cross_entropy(logits, targets):
    """
    logits: [B, C]
    targets: [B]
    """

    log_partition = torch.logsumexp(logits, dim=-1)

    target_logits = logits[
        torch.arange(logits.shape[0]),
        targets,
    ]

    loss = log_partition - target_logits

    return loss.mean()


def test_cross_entropy():
    logits = torch.randn(8, 10)
    targets = torch.randint(0, 10, (8,))

    custom = cross_entropy(logits, targets)

    reference = F.cross_entropy(logits, targets)

    print(custom)
    print(reference)

    print(torch.allclose(custom, reference))


if __name__ == "__main__":
    test_cross_entropy()
