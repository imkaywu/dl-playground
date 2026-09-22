import math


def cosine_lr(
    step,
    warmup_steps,
    total_steps,
    max_lr,
    min_lr=0.0,
):
    """
    Learning-rate schedule:

        1. Linear warmup
        2. Cosine decay

    Args:
        step:
            Current training step.

        warmup_steps:
            Number of steps used for linear warmup.

        total_steps:
            Total number of training steps.

        max_lr:
            Peak learning rate reached after warmup.

        min_lr:
            Final learning rate after cosine decay.

    Returns:
        Learning rate for the given step.
    """

    # ---------------------------------------------------------
    # Phase 1: Linear warmup
    #
    # At step = 0:
    #     lr = 0
    #
    # At step = warmup_steps:
    #     lr = max_lr
    #
    # 0 <= step < warmup_steps
    #
    # lr(step) = max_lr * step / warmup_steps
    # ---------------------------------------------------------
    if step < warmup_steps:
        return max_lr * step / warmup_steps

    # ---------------------------------------------------------
    # Phase 2: Cosine decay
    #
    # First convert the current step into normalized progress.
    #
    # progress = 0.0
    #     -> beginning of cosine decay
    #
    # progress = 1.0
    #     -> end of training
    #
    # warmup_steps <= step <= total_steps
    #
    # progress = (step - warmup_steps) / (total_steps - warmup_steps)
    # ---------------------------------------------------------
    progress = (step - warmup_steps) / (total_steps - warmup_steps)

    # ---------------------------------------------------------
    # Clamp progress to [0, 1].
    #
    # This protects us if step goes beyond total_steps.
    #
    # For example:
    #
    #     step > total_steps
    #         -> progress > 1
    #
    # We don't want the cosine to start increasing again.
    # ---------------------------------------------------------
    progress = min(
        max(progress, 0.0),
        1.0,
    )

    # ---------------------------------------------------------
    # Compute the cosine interpolation factor.
    #
    # progress = 0:
    #     cosine = 1
    #
    # progress = 0.5:
    #     cosine = 0
    #
    # progress = 1:
    #     cosine = -1
    #
    # The 0.5 * (1 + cosine) transformation maps this to:
    #
    # progress = 0   -> 1
    # progress = 0.5 -> 0.5
    # progress = 1   -> 0
    #
    # lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))
    # ---------------------------------------------------------
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

    # ---------------------------------------------------------
    # Interpolate between max_lr and min_lr.
    #
    # At progress = 0:
    #     lr = max_lr
    #
    # At progress = 1:
    #     lr = min_lr
    # ---------------------------------------------------------
    return min_lr + (max_lr - min_lr) * cosine
