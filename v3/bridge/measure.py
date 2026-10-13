"""Score next-token expectations and question answers in the same causal forward."""

import math

import numpy as np
import torch

from v2.experiment import Measurements, patch_state, token_position


class QuestionMeasurements(Measurements):
    def evaluate(
        self,
        text: str,
        site_end: int | None,
        gap_start: int | None,
        layer: int = 17,
        capture: bool = False,
        patch: torch.Tensor | None = None,
    ) -> tuple[dict, torch.Tensor | None, int | None]:
        data = self.encode(text)
        ids = data["inputs"]["input_ids"][0]
        positions = torch.arange(len(ids), device=self.device)
        if gap_start is not None:
            start = next(i for i, (_, end) in enumerate(data["offsets"]) if end > gap_start)
            positions[start:] += 1
        data["inputs"]["position_ids"] = positions[None, :]
        answer_ids = []
        for label in [" A", " B"]:
            full = self.encode(text + label)["inputs"]["input_ids"][0]
            if len(full) != len(ids) + 1 or not torch.equal(full[:-1], ids):
                raise ValueError("Each answer must be exactly one appended token")
            answer_ids.append(int(full[-1]))
        if answer_ids[0] == answer_ids[1]:
            raise ValueError("Answer tokens coincide")
        site = None
        next_ids = []
        if site_end is not None:
            site = token_position(data["offsets"], text.rfind(" ", 0, site_end) + 1, site_end)
            prefix = self.encode(text[:site_end])["inputs"]["input_ids"][0]
            if len(prefix) != site + 1 or not torch.equal(prefix, ids[: site + 1]):
                raise ValueError("Prefix boundary is not token-stable")
            for suffix in [" was", ","]:
                full = self.encode(text[:site_end] + suffix)["inputs"]["input_ids"][0]
                if len(full) != len(prefix) + 1 or not torch.equal(full[:-1], prefix):
                    raise ValueError("Next-token alternative is not one token")
                next_ids.append(int(full[-1]))
        elif capture or patch is not None:
            raise ValueError("Patch and capture require a noun site")
        saved = []
        handle = None
        if capture and patch is not None:
            raise ValueError("Separate capture and patch forwards")
        if capture:

            def save(_module, _inputs, output):
                state = output[0] if isinstance(output, tuple) else output
                saved.append(state[0, site].detach().float().cpu())

            handle = self.layers[layer].register_forward_hook(save)
        if patch is not None:
            handle = self.layers[layer].register_forward_hook(
                lambda _m, _i, output: patch_state(output, site, patch)
            )
        try:
            with torch.inference_mode():
                logits = self.model(**data["inputs"], use_cache=False).logits[0].float()
                answer = logits[-1].log_softmax(-1)[answer_ids].tolist()
                prior = (
                    logits[site].log_softmax(-1)[next_ids].tolist() if site is not None else None
                )
        finally:
            if handle is not None:
                handle.remove()
        if not np.isfinite(answer + (prior or [])).all():
            raise ValueError("Nonfinite score")
        score = dict(
            answer_logp=answer,
            answer_mass=sum(math.exp(v) for v in answer),
            answer_token_ids=answer_ids,
            next_logp=prior,
            next_margin=None if prior is None else prior[0] - prior[1],
            input_ids=ids.tolist(),
            position_ids=positions.tolist(),
            next_token_ids=next_ids,
        )
        return score, saved[0] if saved else None, site
