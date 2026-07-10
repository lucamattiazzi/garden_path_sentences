from dataclasses import dataclass


@dataclass
class ModelSpec:
    hf_name: str
    approx_params: str
    # "float16" is fine for most models, but Gemma is trained in bf16 and fp16
    # inference produces NaN logits — override to "bfloat16" for that family.
    dtype: str = "float16"


MODELS: list[ModelSpec] = [
    ModelSpec("Qwen/Qwen3-0.6B-Base", "0.6B"),
    ModelSpec("Qwen/Qwen3-1.7B-Base", "1.7B"),
    ModelSpec("Qwen/Qwen3-4B-Base", "4B"),
    ModelSpec("Qwen/Qwen3-8B-Base", "8B"),
    ModelSpec("Qwen/Qwen3-14B-Base", "14B"),

    ModelSpec("Qwen/Qwen2.5-3B", "3B"),
    ModelSpec("Qwen/Qwen2.5-7B", "7B"),
    ModelSpec("Qwen/Qwen2.5-14B", "14B"),

    ModelSpec("meta-llama/Llama-3.2-1B", "1B"),
    ModelSpec("meta-llama/Llama-3.2-3B", "3B"),
    ModelSpec("meta-llama/Llama-3.1-8B", "8B"),

    ModelSpec("google/gemma-3-1b-pt", "1B", dtype="bfloat16"),
    ModelSpec("google/gemma-3-4b-pt", "4B", dtype="bfloat16"),
    ModelSpec("google/gemma-3-12b-pt", "12B", dtype="bfloat16"),
    # 27B: dense base, but multimodal wrapper + ~54 GB in bf16 → cloud/GPU only.
    # Not in the original model_finder sweep; run the selection baseline on it
    # before committing (see cloud/ and gemma3_27b/).
    ModelSpec("google/gemma-3-27b-pt", "27B", dtype="bfloat16"),

    ModelSpec("google/gemma-4-E2B", "2.3B eff", dtype="bfloat16"),
    ModelSpec("google/gemma-4-E4B", "4.5B eff", dtype="bfloat16"),
    ModelSpec("openai/gpt-oss-20b", "21B (MoE, ~3.6B active)"),
]
