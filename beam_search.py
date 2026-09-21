import torch


def beam_search(model, start_token, beam_size=3, max_length=10):
    """
    model(sequence) -> logits for next token

    sequence: list[int]
    """

    beams = [([start_token], 0.0)]

    for _ in range(max_length - 1):

        candidates = []

        for sequence, score in beams:

            logits = model(sequence)

            log_probs = torch.log_softmax(logits, dim=-1)

            values, indices = torch.topk(log_probs, beam_size)

            for log_prob, token in zip(values, indices):
                new_sequence = sequence + [token.item()]

                new_score = score + log_prob.item()

                candidates.append((new_sequence, new_score))

        candidates.sort(key=lambda x: x[1], reverse=True)

        beams = candidates[:beam_size]

    return beams


VOCAB_SIZE = 5


def fake_model(sequence):
    return torch.randn(VOCAB_SIZE)


def test_beam_search():

    beams = beam_search(fake_model, start_token=0, beam_size=3, max_length=5)

    for sequence, score in beams:
        print(sequence, score)


if __name__ == "__main__":
    test_beam_search()
