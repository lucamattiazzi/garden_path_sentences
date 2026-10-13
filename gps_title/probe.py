"""Frozen exploratory title examples; token surprisal and elicited interpretation."""

import gc
import hashlib
import importlib.metadata
import json
import re
from datetime import UTC, datetime
from pathlib import Path

CASES = {
    "old": "The old man the boat.",
    "old_clean": "The old people man the boat.",
    "title_exact": "diffusion of language models human migrations",
    "title_article": "The diffusion of language models human migrations.",
    "title_verb_cue": "The diffusion of language actually models human migrations.",
    "title_noun_cue": "The diffusion of language models affects human migrations.",
}
MODELS = {
    "pt": ("google/gemma-3-12b-pt", "295efb63d01a7017928f273a94ebb86105c9526f"),
    "it": ("google/gemma-3-12b-it", "96b6f1eccf38110c56df3a15bffe176da04bfd80"),
}


def word_scores(text, offsets, logps):
    words = list(re.finditer(r"\S+", text))
    result = []
    for word in words:
        hits = [
            i for i, (a, b) in enumerate(offsets) if b > word.start() and a < word.end() and b > a
        ]
        if not hits or any(logps[i] is None for i in hits):
            raise ValueError("Missing word score")
        for i in hits:
            a, b = offsets[i]
            if sum(b > w.start() and a < w.end() for w in words) != 1:
                raise ValueError("Token spans multiple words")
        result.append(
            dict(
                word=word.group(),
                start=word.start(),
                end=word.end(),
                token_indices=hits,
                surprisal_nats=-sum(logps[i] for i in hits),
            )
        )
    return result


def run(output):
    import torch
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).parent
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source.iterdir()
        if p.suffix in (".py", ".md")
    }
    meta = dict(
        complete=False,
        started_at=datetime.now(UTC).isoformat(),
        models=MODELS,
        dtype="float32",
        attention="eager",
        tf32=False,
        gpu=torch.cuda.get_device_name(),
        source_sha256=hashes,
        versions={
            n: importlib.metadata.version(n) for n in ("torch", "transformers", "tokenizers")
        },
    )
    (output / "source.json").write_text(
        json.dumps({p.name: p.read_text() for p in source.iterdir() if p.name in hashes}, indent=2)
    )
    (output / "metadata.json").write_text(json.dumps(meta, indent=2))
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = dict(
        design="Exploratory fixed examples; no circuit intervention; not a generalization test",
        surprisal={},
        interpretations={},
    )
    try:
        for variant in ("pt", "it"):
            model_id, revision = MODELS[variant]
            tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
            model, info = Gemma3ForConditionalGeneration.from_pretrained(
                model_id,
                revision=revision,
                dtype=torch.float32,
                attn_implementation="eager",
                output_loading_info=True,
            )
            if any(
                info.get(k)
                for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
            ):
                raise ValueError("Checkpoint loading mismatch: " + str(info))
            model = model.to("cuda").eval()
            for name, text in CASES.items():
                if variant == "pt":
                    encoded = tok(text, return_tensors="pt", return_offsets_mapping=True)
                    offsets = encoded.pop("offset_mapping")[0].tolist()
                    inputs = {k: v.to("cuda") for k, v in encoded.items()}
                    ids = inputs["input_ids"][0]
                    with torch.inference_mode():
                        logits = model(**inputs, use_cache=False).logits[0].float()
                        logprobs = logits.log_softmax(-1)
                        lp = [None] + logprobs[:-1].gather(
                            1, ids[1:, None]
                        ).flatten().cpu().tolist()
                    words = word_scores(text, offsets, lp)
                    for w in words:
                        first = w["token_indices"][0]
                        top = logprobs[first - 1].topk(5)
                        w["top5_before_word"] = [
                            dict(token_id=i, piece=tok.decode([i]), log_probability=v)
                            for i, v in zip(
                                top.indices.cpu().tolist(), top.values.cpu().tolist(), strict=True
                            )
                        ]
                    result["surprisal"][name] = dict(
                        text=text,
                        token_ids=ids.cpu().tolist(),
                        pieces=tok.convert_ids_to_tokens(ids.cpu().tolist()),
                        offsets=offsets,
                        token_logps=lp,
                        words=words,
                    )
                    del logits, logprobs, inputs, ids
                else:
                    question = (
                        "Read this English sentence. Paraphrase it in one sentence, then quote its grammatical subject and its main verb. If it is incomplete or ungrammatical, explain why.\n\n"
                        + text
                    )
                    prompt = tok.apply_chat_template(
                        [dict(role="user", content=question)],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    inputs = tok(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
                    with torch.inference_mode():
                        generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)
                    result["interpretations"][name] = dict(
                        text=text,
                        prompt=prompt,
                        response=tok.decode(
                            generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
                        ),
                        max_new_tokens=256,
                        generated_tokens=int(generated.shape[1] - inputs["input_ids"].shape[1]),
                    )
                    del inputs, generated
                (output / "results.json").write_text(
                    json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
                )
            del model, tok
            gc.collect()
            torch.cuda.empty_cache()
        meta["complete"] = True
    finally:
        meta["finished_at"] = datetime.now(UTC).isoformat()
        (output / "metadata.json").write_text(json.dumps(meta, indent=2))
