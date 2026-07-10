"""Model-structure accessors that work across two layouts:

  * flat decoder-only models — Qwen3, Qwen2.5, Llama-3, and the text-only
    Gemma-3-1B (`Gemma3ForCausalLM`): the decoder is `model.model.layers`
    and the head counts are on `model.config` directly;
  * nested multimodal wrappers — Gemma-3 4B/12B/27B: the language model and
    its config sit one level deeper (`…language_model.layers`,
    `config.text_config`).

The old code hard-coded `model.model.layers` and `model.config`. These helpers
resolve to *exactly those objects* for the flat models, so the Qwen/Llama
pipeline is byte-for-byte unchanged; they only add the extra lookup paths that
the Gemma multimodal checkpoints need.
"""

from __future__ import annotations


def text_config(model):
    """Config object carrying num_hidden_layers / num_attention_heads / head_dim.

    Multimodal Gemma-3 nests the text params under `config.text_config`; flat
    models expose them directly, in which case `getattr` falls back to `config`.
    """
    cfg = model.config
    return getattr(cfg, "text_config", None) or cfg


# Attribute paths tried in order; the first one that yields a non-empty list of
# blocks exposing `.self_attn.o_proj` wins. Ordered flat-first so the common
# case resolves immediately.
_LAYER_PATHS = (
    "model.layers",                 # Qwen / Llama / Gemma3ForCausalLM (1B)
    "model.language_model.layers",  # Gemma-3 multimodal (Gemma3Model)
    "language_model.model.layers",  # alternative wrapper nesting
    "model.model.layers",           # defensive extra nesting
    "layers",                       # a bare decoder passed directly
)


def _resolve(obj, dotted: str):
    for name in dotted.split("."):
        obj = getattr(obj, name, None)
        if obj is None:
            return None
    return obj


def decoder_layers(model):
    """Return the ModuleList of transformer decoder blocks, whatever the
    wrapper nesting is. Each block is expected to expose `.self_attn.o_proj`
    (true for Qwen, Llama and Gemma-3 — all standard attention with an output
    projection, unlike the Qwen3.5 Gated-DeltaNet hybrid)."""
    for path in _LAYER_PATHS:
        layers = _resolve(model, path)
        if (
            layers is not None
            and len(layers) > 0
            and hasattr(layers[0], "self_attn")
            and hasattr(layers[0].self_attn, "o_proj")
        ):
            return layers
    raise AttributeError(
        f"arch.decoder_layers: no decoder layer list with .self_attn.o_proj on "
        f"{type(model).__name__}. Add its attribute path to arch._LAYER_PATHS."
    )
