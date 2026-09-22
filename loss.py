import torch
import torch.nn.functional as F


def cross_entropy(logits, targets):
    """
    logits: [B, C]
    targets: [B]
    """

    log_partition = torch.logsumexp(logits, dim=-1)

    target_logits = logits[
        torch.arange(logits.shape[0]),
        targets,
    ]

    loss = log_partition - target_logits

    return loss.mean()


def test_cross_entropy():
    logits = torch.randn(8, 10)
    targets = torch.randint(0, 10, (8,))

    custom = cross_entropy(logits, targets)

    reference = F.cross_entropy(logits, targets)

    print(custom)
    print(reference)

    print(torch.allclose(custom, reference))


def contrastive_loss(
    z1,
    z2,
    temperature=0.07,
):
    """
    Batch contrastive loss (InfoNCE / CLIP-style).

    z1, z2:
        [B, D]
        B = batch size
        D = embedding dimension

        z1[i] and z2[i] are a positive pair.

        All other combinations in the batch are treated
        as negative pairs.

    Example with B=4:

                 z2
              0    1    2    3
           +----+----+----+----+
    z1  0  | +  | -  | -  | -  |
        1  | -  | +  | -  | -  |
        2  | -  | -  | +  | -  |
        3  | -  | -  | -  | +  |
           +----+----+----+----+
    """

    # ---------------------------------------------------------
    # 1. Normalize each embedding to unit length.
    #
    # After normalization:
    #
    #     z1[i] · z2[j]
    #
    # is the cosine similarity between the two embeddings.
    #
    # Shape stays:
    #     z1: [B, D]
    #     z2: [B, D]
    # ---------------------------------------------------------
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    # ---------------------------------------------------------
    # 2. Compute similarity between EVERY z1 and EVERY z2.
    #
    # z1 @ z2.T:
    #
    #     [B, D] @ [D, B] -> [B, B]
    #
    # logits[i, j] = similarity(z1[i], z2[j])
    #
    # The diagonal contains the positive pairs:
    #
    #     logits[0, 0] -> positive
    #     logits[1, 1] -> positive
    #     logits[2, 2] -> positive
    #
    # Everything off the diagonal is a negative pair.
    # ---------------------------------------------------------
    logits = z1 @ z2.T

    # ---------------------------------------------------------
    # 3. Temperature scaling.
    #
    # Smaller temperature makes the softmax distribution
    # sharper, so the model is penalized more strongly when
    # the positive similarity is not clearly higher than
    # the negative similarities.
    #
    # Example:
    #
    #     similarity = 0.8
    #     temperature = 0.1
    #
    #     logit = 0.8 / 0.1 = 8.0
    # ---------------------------------------------------------
    logits = logits / temperature

    # ---------------------------------------------------------
    # 4. Create the correct class for each row.
    #
    # If B = 4:
    #
    #     labels = [0, 1, 2, 3]
    #
    # This tells cross_entropy:
    #
    #     row 0 -> correct answer is column 0
    #     row 1 -> correct answer is column 1
    #     row 2 -> correct answer is column 2
    #     row 3 -> correct answer is column 3
    #
    # In other words, for each z1[i], we want to find
    # its matching z2[i].
    # ---------------------------------------------------------
    labels = torch.arange(
        z1.shape[0],
        device=z1.device,
    )

    # ---------------------------------------------------------
    # 5. z1 -> z2 contrastive loss.
    #
    # For each z1[i], cross_entropy asks:
    #
    #     "Which z2[j] is the matching embedding?"
    #
    # The correct answer is j = i.
    #
    # So this encourages:
    #
    #     similarity(z1[i], z2[i])
    #
    # to be larger than:
    #
    #     similarity(z1[i], z2[j])  for j != i
    # ---------------------------------------------------------
    loss_12 = F.cross_entropy(
        logits,
        labels,
    )

    # ---------------------------------------------------------
    # 6. z2 -> z1 contrastive loss.
    #
    # We now reverse the direction:
    #
    #     "For each z2[i], which z1[j] is its match?"
    #
    # Transposing logits swaps rows and columns:
    #
    #     logits.T[j, i] = logits[i, j]
    #
    # This makes each z2 the query and each z1 a candidate.
    # ---------------------------------------------------------
    loss_21 = F.cross_entropy(
        logits.T,
        labels,
    )

    # ---------------------------------------------------------
    # 7. Symmetric contrastive loss.
    #
    # We want both directions to work:
    #
    #     z1 -> z2
    #     z2 -> z1
    #
    # So we simply average the two losses.
    # ---------------------------------------------------------
    return 0.5 * (loss_12 + loss_21)


if __name__ == "__main__":
    test_cross_entropy()
