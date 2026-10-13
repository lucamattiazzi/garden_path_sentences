"""Full answer-sequence likelihood, with a stable shared prompt boundary."""

import torch

from v2.experiment import Measurements


def sequence_logps(logits: torch.Tensor, ids: torch.Tensor, start: int) -> list[float]:
    if not 0 < start < len(ids):
        raise ValueError("A nonempty suffix and prefix are required")
    logps = logits[start - 1 : -1].float().log_softmax(-1)
    return logps.gather(-1, ids[start:, None]).squeeze(-1).tolist()


class DirectMeasurements(Measurements):
    def score(self, text: str, candidates: list[str]) -> dict:
        prefix = self.encode(text)["inputs"]["input_ids"][0]
        scores = []
        for suffix in candidates:
            inputs = self.encode(text + suffix)["inputs"]
            ids = inputs["input_ids"][0]
            if not torch.equal(ids[: len(prefix)], prefix) or len(ids) <= len(prefix):
                raise ValueError("Answer boundary is not token stable")
            with torch.inference_mode():
                logits = self.model(**inputs, use_cache=False).logits[0]
                values = sequence_logps(logits, ids, len(prefix))
            if not all(torch.isfinite(torch.tensor(values))):
                raise ValueError("Nonfinite scores")
            scores.append(
                dict(token_ids=ids[len(prefix) :].tolist(), token_logp=values, logp=sum(values))
            )
        return dict(prefix_ids=prefix.tolist(), candidates=scores)
