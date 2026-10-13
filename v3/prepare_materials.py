"""Reproduce the frozen NP/Z subset from the authors' pinned public data."""

import csv
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen

COMMIT = "3ea8380a08e6ea601cf65f1b624f74ef29fc765b"
BASE = f"https://raw.githubusercontent.com/hannamw/GP-mechanisms/{COMMIT}/data_csv/"
FILES = {
    "gp_same_len.csv": "b6a5dd3a9c8550328d55272107e68fcb5d81887382798d978d178b5460537c7b",
    "gp_orig.csv": "079ec2e5d3456d40fa26c9fd3c8a345fef6b47f3204040c0f4062184a937ff26",
}


def build_materials(same: str, original: str) -> list[dict]:
    rows = [r for r in csv.DictReader(io.StringIO(same)) if r["condition"] == "NPZ"]
    full = {
        r["item"]: r
        for r in csv.DictReader(io.StringIO(original))
        if r["condition"] == "NPZ_UAMB" and r["ambiguity"] == "ambiguous"
    }
    verbs = sorted(
        {r["sentence_ambiguous"].split()[-3] for r in rows},
        key=lambda v: hashlib.sha256(("gps-v3-split:" + v).encode()).hexdigest(),
    )
    development = set(verbs[:6])
    items = []
    for r in rows:
        prefix = r["sentence_ambiguous"]
        words = prefix.split()
        target, verb, actor = words[-1], words[-3], words[2]
        # The original file contains modifiers absent in the equal-length variant.
        # Extract the suffix after its target noun rather than reuse its word index.
        marker = f" the {target} "
        sentence = full[r["item"]]["Sentence"]
        if sentence.count(marker) != 1 or words[-2] != "the":
            raise ValueError("Ambiguous source alignment")
        continuation = " ".join(sentence.split(marker)[1].split()[:3])
        prefixes = {"optional": prefix, "intransitive": r["sentence_post"]}
        paired = next(
            other
            for other in rows
            if other["item"] != r["item"] and other["sentence_ambiguous"].split()[-3] == verb
        )
        other_actor = paired["sentence_ambiguous"].split()[2]
        items.append(
            dict(
                id="npz-" + r["item"],
                source_item=int(r["item"]),
                family=verb,
                split="development" if verb in development else "test",
                actor=actor,
                other_actor=other_actor,
                target=target,
                prefixes=prefixes,
                continuation=continuation,
                original_ambiguous_sentence=sentence,
            )
        )
    return items


def main() -> None:
    source = {}
    for name, expected in FILES.items():
        content = urlopen(BASE + name, timeout=30).read()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Changed source: {name}")
        source[name] = content.decode()
    payload = dict(
        source=dict(
            repository="https://github.com/hannamw/GP-mechanisms",
            commit=COMMIT,
            files=FILES,
            paper="https://aclanthology.org/2025.naacl-long.164/",
            license="No explicit repository license found; no new license asserted.",
        ),
        transformation="24 NP/Z equal-word-length prefixes; optional/intransitive variants; first three words of original main clause; SHA256-ordered optional-verb split",
        items=build_materials(source["gp_same_len.csv"], source["gp_orig.csv"]),
    )
    Path(__file__).with_name("materials").joinpath("npz.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
