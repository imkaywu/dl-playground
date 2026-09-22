import numpy as np


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)

    exp_x = np.exp(x)

    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def softmax_backward(x, grad_output):
    """
    x:            [B, C]
    grad_output:  [B, C] = dL/dp

    Returns:
        grad_x: [B, C] = dL/dx
    """

    p = softmax(x)

    # Derive the softmax Jacobian:
    #
    #   p_i = exp(x_i) / sum_k exp(x_k)
    #
    # For i == j, both numerator and denominator depend on x_i:
    #
    #   dp_i/dx_i
    #       = p_i (1 - p_i)
    #
    # For i != j, only the denominator depends on x_j:
    #
    #   dp_i/dx_j
    #       = -p_i p_j
    #
    # Therefore:
    #
    #   dp_i/dx_j
    #       = p_i (delta_ij - p_j)
    #
    # where delta_ij = 1 if i == j, else 0.
    #
    # Now apply the chain rule:
    #
    #   dL/dx_j
    #       = sum_i (dL/dp_i) (dp_i/dx_j)
    #
    #       = sum_i grad_output_i * p_i (delta_ij - p_j)
    #
    #       = p_j [grad_output_j - sum_i p_i * grad_output_i]
    #
    # Thus:
    #
    #   grad_x = p * (grad_output - sum(p * grad_output))

    dot = np.sum(
        grad_output * p,
        axis=-1,
        keepdims=True,
    )

    grad_x = p * (grad_output - dot)

    return grad_x


def test_softmax():
    x = np.random.randn(4, 5)
    grad = np.random.randn(4, 5)

    y = softmax(x)
    dx = softmax_backward(x, grad)

    print(y.shape)
    print(dx.shape)
    print(y.sum(axis=-1))


if __name__ == "__main__":
    test_softmax()
