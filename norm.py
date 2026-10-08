import numpy as np
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()

        # Learnable scale (gamma) and bias (beta), each with shape [D]
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

        self.eps = eps

    def forward(self, x):
        """
        LayerNorm normalizes the D features of each token
        independently.
        """
        # Calculate the mean of each token's D features.
        #
        # mu = (1 / D) * sum(x_i), for i =1, ..., D
        #
        # shape: [..., D] -> [..., 1]
        mean = x.mean(dim=-1, keepdim=True)

        # Calculate the population variance.
        #
        # var = (1 / D) sum((x_i - mu)^2))
        #
        # shape: [..., D] -> [..., 1]
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)

        # Normalize each feature to approximately zero mean and unit
        # variance.
        #
        # x_hat_i = (x_i - mu) / (sqrt(var + eps))
        #
        # eps prevents division by zero
        # shape: [..., D]
        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        # Apply learnable scale and bias.
        #
        # y_i = gamma_i * x_hat_i + beta_i
        #
        # gamma and beta have shape [D] and broadcast across B and T.
        # shape: [..., D]
        return self.gamma * x_hat + self.beta


def test_layer_norm():
    B, T, D = 2, 5, 32
    x = torch.randn(B, T, D)

    ln = LayerNorm(d_model=D)
    y = ln(x)

    assert y.shape == x.shape

    torch_ln = nn.LayerNorm(D)
    y2 = torch_ln(x)

    assert y2.shape == x.shape

    print(torch.allclose(y, y2, atol=1e-5))
    print("LayerNorm test passed: ", y.shape)


class BatchNorm:
    def __init__(self, D, eps=1e-5):
        self.gamma = np.ones(D)
        self.beta = np.zeros(D)
        self.eps = eps

        self.cache = None

        # Gradients
        self.dgamma = None
        self.dbeta = None

    def forward(self, x):
        """
        x: [B, D]
        """

        mean = x.mean(axis=0)
        var = x.var(axis=0)

        x_hat = (x - mean) / np.sqrt(var + self.eps)

        out = self.gamma * x_hat + self.beta

        self.cache = (
            x,
            x_hat,
            mean,
            var,
        )

        return out

    def backward(self, dout):
        """
        dout: [B, D]
        """

        x, x_hat, mean, var = self.cache

        B, D = x.shape

        x_centered = x - mean

        std_inv = 1.0 / np.sqrt(var + self.eps)

        self.dgamma = np.sum(
            dout * x_hat,
            axis=0,
        )

        self.dbeta = np.sum(
            dout,
            axis=0,
        )

        dx_hat = dout * self.gamma

        dvar = (
            np.sum(
                dx_hat * x_centered,
                axis=0,
            )
            * (-0.5)
            * std_inv**3
        )

        dmean = np.sum(
            dx_hat * -std_inv,
            axis=0,
        ) + dvar * np.mean(-2 * x_centered, axis=0)

        dx = dx_hat * std_inv + dvar * 2 * x_centered / B + dmean / B

        return dx


def test_batch_norm():
    B = 4
    D = 3

    x = np.random.randn(B, D)

    bn = BatchNorm(D)

    out = bn.forward(x)

    dout = np.random.randn(B, D)
    dx = bn.backward(dout)

    print("out:", out.shape)  # (4, 3)
    print("dx:", dx.shape)  # (4, 3)
    print("dgamma:", bn.dgamma)  # (3,)
    print("dbeta:", bn.dbeta)  # (3,)


if __name__ == "__main__":
    test_layer_norm()

    test_batch_norm()
