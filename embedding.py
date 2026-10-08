import math

import torch
import torch.nn as nn

from pe import SinusoidalPositionalEncoding


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, model_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, model_dim)

    def forward(self, token_ids):
        """
        token_ids: [B, T], integer token IDs
        returns: [B, T, D]
        """

        return self.embedding(token_ids)


class InputEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size,
        model_dim,
        max_len=5000,
    ):
        super().__init__()

        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size, model_dim=model_dim
        )

        self.positional_encoding = SinusoidalPositionalEncoding(
            model_dim=model_dim, max_len=max_len
        )

        self.embedding_scale = math.sqrt(model_dim)

    def forward(self, token_ids):
        """
        token_ids: [B, T]
        returns: [B, T, D]
        """
        x = self.token_embedding(token_ids) * self.embedding_scale
        x = self.positional_encoding(x)

        return x


def test_input_embedding():
    B, T = 2, 10
    vocab_size = 1000
    model_dim = 64

    token_ids = torch.randint(0, vocab_size, (B, T))

    embedding = InputEmbedding(
        vocab_size=vocab_size,
        model_dim=model_dim,
        max_len=100,
    )

    x = embedding(token_ids)

    assert x.shape == (B, T, model_dim)
    print("Input embedding test passed:", x.shape)


if __name__ == "__main__":
    test_input_embedding()
