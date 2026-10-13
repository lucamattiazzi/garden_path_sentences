"""Read exact next-token alternatives and per-token continuation probabilities."""

import re

import numpy as np
import torch

from v2.experiment import Measurements, patch_state


class TokenMeasurements(Measurements):
    def inputs(self, prefix: str, suffix: str = "", gap: bool = False) -> tuple[dict, int]:
        base = self.encode(prefix)["inputs"]["input_ids"][0]
        data = self.encode(prefix + (" " + suffix if suffix else ""))
        ids = data["inputs"]["input_ids"][0]
        if not torch.equal(ids[: len(base)], base):
            raise ValueError("Continuation changes prefix tokenization")
        positions = torch.arange(len(ids), device=self.device)
        if gap:
            boundary = prefix.rindex(" the ") + 1
            first = next(i for i, (_, end) in enumerate(data["offsets"]) if end > boundary)
            positions[first:] += 1
        data["inputs"]["position_ids"] = positions[None, :]
        return data, len(base)

    def validate_alignment(self, native: str, comma: str) -> None:
        plain, _ = self.inputs(native, gap=True)
        marked, _ = self.inputs(comma)
        comma_char = comma.index(",")
        hits = [i for i, (start, end) in enumerate(marked["offsets"]) if start <= comma_char < end]
        if len(hits) != 1:
            raise ValueError("Comma must be a single separate token")
        keep = [i for i in range(len(marked["offsets"])) if i not in hits]
        for key in ["input_ids", "position_ids"]:
            if not torch.equal(marked["inputs"][key][:, keep], plain["inputs"][key]):
                raise ValueError(f"Comma alignment failed: {key}")

    def evaluate(
        self,
        prefix: str,
        suffix: str = "",
        gap: bool = False,
        patch: tuple | None = None,
        capture: bool = False,
    ) -> tuple[dict, list, int]:
        data, length = self.inputs(prefix, suffix, gap)
        ids = data["inputs"]["input_ids"][0]
        alternatives = []
        for ending in [" was", ","]:
            full = self.encode(prefix + ending)["inputs"]["input_ids"][0]
            if len(full) != length + 1 or not torch.equal(full[:length], ids[:length]):
                raise ValueError("Next-token alternatives must each be exactly one appended token")
            alternatives.append(int(full[-1]))
        if alternatives[0] == alternatives[1]:
            raise ValueError("Alternatives must be different tokens")
        states, handles = [], []
        position = length - 1
        if capture:

            def save(_module, _inputs, output):
                state = output[0] if isinstance(output, tuple) else output
                states.append(state[0, position].detach().float().cpu())

            handles += [layer.register_forward_hook(save) for layer in self.layers]
        if patch:
            layer, where, vector = patch
            handles.append(
                self.layers[layer].register_forward_hook(
                    lambda _m, _i, output: patch_state(output, where, vector)
                )
            )
        try:
            with torch.inference_mode():
                logits = self.model(**data["inputs"], use_cache=False).logits[0].float()
                next_logp = logits[position].log_softmax(-1)[alternatives].tolist()
                indices = torch.arange(length, len(ids), device=self.device)
                logp = (
                    logits[indices - 1]
                    .log_softmax(-1)
                    .gather(1, ids[indices, None])
                    .flatten()
                    .tolist()
                )
        finally:
            for handle in handles:
                handle.remove()
        if not np.isfinite(next_logp + logp).all():
            raise ValueError("Nonfinite measurement")
        offsets = data["offsets"][length:]
        words, word_logp = [], []
        used = []
        for word in re.finditer(r"\S+", suffix):
            start, end = len(prefix) + 1 + word.start(), len(prefix) + 1 + word.end()
            hits = [i for i, (a, b) in enumerate(offsets) if b > start and a < end]
            if not hits:
                raise ValueError("Word has no aligned tokens")
            words.append(word.group())
            word_logp.append(sum(logp[i] for i in hits))
            used += hits
        if sorted(used) != list(range(len(logp))):
            raise ValueError("Continuation tokens do not partition into words")
        result = dict(
            was_logp=next_logp[0],
            comma_logp=next_logp[1],
            margin=next_logp[0] - next_logp[1],
            alternative_token_ids=alternatives,
            token_logp=logp,
            words=words,
            word_logp=word_logp,
            token_ids=ids[length:].tolist(),
            token_pieces=self.tokenizer.convert_ids_to_tokens(ids[length:].tolist()),
            input_ids=ids.tolist(),
            position_ids=data["inputs"]["position_ids"][0].tolist(),
        )
        return result, states, position
