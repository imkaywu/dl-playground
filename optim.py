import torch


def sgd_step(params, lr):
    with torch.no_grad():

        for p in params:
            p -= lr * p.grad

            p.grad = None


class Adam:
    def __init__(
        self,
        params,
        lr=1e-3,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
    ):
        self.params = list(params)

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = [torch.zeros_like(p) for p in self.params]

        self.v = [torch.zeros_like(p) for p in self.params]

        self.t = 0

    def step(self):
        self.t += 1

        with torch.no_grad():

            for i, p in enumerate(self.params):

                g = p.grad

                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g

                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g

                m_hat = self.m[i] / (1 - self.beta1**self.t)

                v_hat = self.v[i] / (1 - self.beta2**self.t)

                p -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

                p.grad = None
