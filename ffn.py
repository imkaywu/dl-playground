import torch
import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        # Expand the feature dimension
        self.fc1 = nn.Linear(d_model, d_ff)

        # Introduce nonlinearity
        self.activation = nn.GELU()

        # Project back to the original model dimension.
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """
        x: [B, T, D]
        returns: [B, T, D]
        """

        # [B, T, D] -> [D, T, D_ff]
        x = self.fc1(x)

        x = self.activation(x)

        # [B, T, D_ff] -> [B, T, D]
        x = self.fc2(x)

        return x


def test_feed_forward_network():
    B, T, D = 2, 5, 32
    D_ff = 128

    x = torch.randn(B, T, D)

    ffn = FeedForwardNetwork(d_model=D, d_ff=D_ff)
    y = ffn(x)

    assert y.shape == (B, T, D)
    print("FFN test passed:", y.shape)


if __name__ == "__main__":
    test_feed_forward_network()
