import numpy as np
import torch
import torch.nn as nn


class MyLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)

        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)

        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        return self.gamma * x_hat + self.beta


def test_layer_norm():
    x = torch.randn(2, 5, 8)

    my_ln = MyLayerNorm(8)
    torch_ln = nn.LayerNorm(8)

    y1 = my_ln(x)
    y2 = torch_ln(x)

    print(torch.allclose(y1, y2, atol=1e-5))


class MyBatchNorm:
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

    bn = MyBatchNorm(D)

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
