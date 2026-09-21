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


if __name__ == "__main__":
    test_layer_norm()
