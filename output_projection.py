import torch
import torch.nn as nn


class OutputProjection(nn.Module):
    def __init__(self, model_dim, vocab_size):
        super().__init__()

        # Map each token's contextual vector to vocabulary scores
        self.proj = nn.Linear(model_dim, vocab_size)

    def forward(self, x):
        """
        x: [B, T, D]
        returns: [B, T, V] - raw vocabulary logits
        """
        return self.proj(x)


def test_output_projection():
    B, T, D = 2, 10, 64
    vocab_size = 1000

    hidden = torch.randn(B, T, D)
    targets = torch.randint(0, vocab_size, (B, T))

    output_proj = OutputProjection(D, vocab_size)
    logits = output_proj(hidden)  # [B, T, V]

    loss_fn = nn.CrossEntropyLoss()

    # CrossEntropyLoss expects classes in dimension 1,
    # so move V before T and B.
    loss = loss_fn(
        logits.reshape(B * T, vocab_size),
        targets.reshape(B * T),
    )

    print("Logits:", logits.shape)
    print("Loss:", loss.item())
