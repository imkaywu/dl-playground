import torch


def conv2d(x, weight, bias=None):
    """
    x:
        [B, C_in, H, W]

    weight:
        [C_out, C_in, K, K]

    returns:
        [B, C_out, H-K+1, W-K+1]
    """

    B, C_in, H, W = x.shape
    C_out, _, K, _ = weight.shape

    out_h = H - K + 1
    out_w = W - K + 1

    out = torch.empty(
        B,
        C_out,
        out_h,
        out_w,
    )

    for b in range(B):
        for c_out in range(C_out):
            for i in range(out_h):
                for j in range(out_w):

                    patch = x[
                        b,
                        :,
                        i : i + K,
                        j : j + K,
                    ]

                    out[
                        b,
                        c_out,
                        i,
                        j,
                    ] = (patch * weight[c_out]).sum()

                    if bias is not None:
                        out[
                            b,
                            c_out,
                            i,
                            j,
                        ] += bias[c_out]

    return out
