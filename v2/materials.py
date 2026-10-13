"""Synthetic exploratory materials; not human-validated or a confirmatory set."""

from __future__ import annotations


def garden_paths() -> list[dict]:
    families = [
        ("washed", "mother", "baby", "sleeping", "father"),
        ("dressed", "parent", "child", "smiling", "nanny"),
        ("hunted", "hunter", "deer", "running", "ranger"),
        ("watched", "guard", "prisoner", "waiting", "officer"),
        ("visited", "doctor", "patient", "resting", "nurse"),
        ("called", "teacher", "student", "listening", "principal"),
        ("trained", "coach", "athlete", "stretching", "instructor"),
        ("read", "editor", "report", "missing", "author"),
    ]
    return [
        {
            "id": verb,
            "family": verb,
            "ambiguous": f"While the {actor} {verb} the {target}",
            "clean": f"While the {actor} {verb}, the {target}",
            "lexical_ambiguous": f"While the {other} {verb} the {target}",
            "lexical_clean": f"While the {other} {verb}, the {target}",
            "critical": "was",
            "spillover": spillover,
        }
        for verb, actor, target, spillover, other in families
    ]
