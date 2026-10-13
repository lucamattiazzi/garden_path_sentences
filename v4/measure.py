"""V4 endpoints with equal forward shapes for unequal answer lengths."""

import torch

from v2.experiment import Measurements, patch_state, token_position
from v3.direct.measure import sequence_logps


class BridgeMeasurements(Measurements):
    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer
        self.layers = getattr(model.model, "language_model", model.model).layers
        self.device = next(model.parameters()).device

    def evaluate(self, p: dict, layer: int, capture: bool = False, donor=None):
        base = self.encode(p["text"])
        prefix = base["inputs"]["input_ids"][0]
        site = None
        if p["site_end"] is not None:
            end = p["site_end"]
            site = token_position(base["offsets"], p["text"].rfind(" ", 0, end) + 1, end)
            noun = self.encode(p["text"][:end])["inputs"]["input_ids"][0]
            if len(noun) != site + 1 or not torch.equal(noun, prefix[: site + 1]):
                raise ValueError("Unstable noun boundary")
            next_ids = []
            for suffix in [" was", ","]:
                encoded = self.encode(p["text"][:end] + suffix)["inputs"]["input_ids"][0]
                if len(encoded) != len(noun) + 1 or not torch.equal(encoded[:-1], noun):
                    raise ValueError("Next alternatives must each be one token")
                next_ids.append(int(encoded[-1]))
        if (capture or donor is not None) and site is None:
            raise ValueError("Intervention requires a noun site")
        if capture and donor is not None:
            raise ValueError("Separate capture and patch calls")
        answers, states, nexts, details = [], [], [], []
        encoded = [self.encode(p["text"] + suffix) for suffix in p["candidates"]]
        width = max(d["inputs"]["input_ids"].shape[1] for d in encoded)
        for data in encoded:
            ids = data["inputs"]["input_ids"][0]
            if not torch.equal(ids[: len(prefix)], prefix) or len(ids) <= len(prefix):
                raise ValueError("Unstable answer boundary")
            positions = torch.arange(len(ids), device=self.device)
            if p["gap_start"] is not None:
                start = next(
                    i for i, (_, end) in enumerate(data["offsets"]) if end > p["gap_start"]
                )
                positions[start:] += 1
            # Right padding keeps GEMM shapes identical; never score padding tokens.
            padding = width - len(ids)
            for key in ["input_ids", "attention_mask"]:
                data["inputs"][key] = torch.nn.functional.pad(
                    data["inputs"][key], (0, padding), value=0
                )
            full_positions = torch.arange(width, device=self.device)
            full_positions[: len(ids)] = positions
            data["inputs"]["position_ids"] = full_positions[None, :]
            handle = None
            if capture:

                def save(_m, _i, out):
                    state = out[0] if isinstance(out, tuple) else out
                    states.append(state[0, site].detach().float().cpu())

                handle = self.layers[layer].register_forward_hook(save)
            elif donor is not None:
                handle = self.layers[layer].register_forward_hook(
                    lambda _m, _i, out: patch_state(out, site, donor)
                )
            try:
                with torch.inference_mode():
                    logits = (
                        self.model(**data["inputs"], use_cache=False).logits[0, : len(ids)].float()
                    )
                    token_logs = sequence_logps(logits, ids, len(prefix))
                    next_logs = (
                        logits[site].log_softmax(-1)[next_ids].tolist()
                        if site is not None
                        else None
                    )
            finally:
                if handle is not None:
                    handle.remove()
            answers.append(sum(token_logs))
            details.append(dict(token_ids=ids[len(prefix) :].tolist(), token_logp=token_logs))
            nexts.append(next_logs)
        if capture:
            torch.testing.assert_close(states[0], states[1], rtol=0, atol=1e-5)
        if (
            site is not None
            and max(abs(a - b) for a, b in zip(nexts[0], nexts[1], strict=True)) > 1e-5
        ):
            raise ValueError("Future answer changed earlier logits")
        if not torch.isfinite(torch.tensor(answers + (nexts[0] or []))).all():
            raise ValueError("Nonfinite endpoint")
        return dict(
            answer_logp=answers,
            answer_tokens=details,
            next_logp=nexts[0],
            next_margin=None if site is None else nexts[0][0] - nexts[0][1],
            input_ids=prefix.tolist(),
            position_ids=positions[: len(prefix)].tolist(),
            site=site,
        ), states[0] if capture else None
