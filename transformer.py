import torch
import torch.nn as nn

from embedding import InputEmbedding
from encoder import Encoder
from output_projection import OutputProjection


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        model_dim,
        num_heads,
        ff_dim,
        num_layers,
        max_len=5000,
    ):
        super().__init__()

        self.input_embedding = InputEmbedding(
            vocab_size=vocab_size, model_dim=model_dim, max_len=max_len
        )

        self.encoder = Encoder(
            model_dim=model_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
        )

        self.output_projection = OutputProjection(
            model_dim=model_dim, vocab_size=vocab_size
        )

    def forward(self, token_ids, mask=None):
        """
        token_ids: [B, T]
        mask: optional attention mask

        returns: [B, T, vocab_size]
        """

        # Token IDs -> embedding + positional encoding
        x = self.input_embedding(token_ids)

        # Build contextual representation
        x = self.encoder(x, mask=mask)

        # Convert each position into vocabulary logits
        logits = self.output_projection(x)

        return logits


def test_transformer():
    B, T = 2, 10
    vocab_size = 1000
    model_dim = 64

    model = Transformer(
        vocab_size=vocab_size,
        model_dim=model_dim,
        num_heads=4,
        ff_dim=256,
        num_layers=2,
        max_len=100,
    )

    # Random token IDs.
    token_ids = torch.randint(0, vocab_size, (B, T))

    # Random target token IDs.
    targets = torch.randint(0, vocab_size, (B, T))

    # Forward pass.
    logits = model(token_ids)

    assert logits.shape == (B, T, vocab_size)

    # CrossEntropyLoss expects logits shaped [N, C]
    # and targets shaped [N], where C is the number of classes.
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(
        logits.reshape(B * T, vocab_size),
        targets.reshape(B * T),
    )

    # Backpropagation test.
    loss.backward()

    print("Logits shape:", logits.shape)
    print("Loss:", loss.item())
    print("Transformer forward/backward test passed.")


if __name__ == "__main__":
    test_transformer()
