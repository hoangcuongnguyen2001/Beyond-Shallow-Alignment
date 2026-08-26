"""
Phase 2: Component-level analysis via attribution patching.

Run after activation_patching.py identifies peak causal layers per paradigm.

Attribution patching ranks components with:
    grad_metric(corrupted_component) * (clean_component - corrupted_component)

This script follows the current activation_patching.py contract:
  - prompt pairs are filtered with validate_pairs_generative()
  - refusal/compliance token ids are per valid pair
  - metrics are read at the last meaningful prompt position, not blindly at -1

Usage:
    python component_analysis.py \
        --model_path <hf_path> \
        --model_name sft \
        --arch llama \
        --harmful_path refusal_direction/dataset/splits/harmful_train.json \
        --harmless_path refusal_direction/dataset/splits/harmless_train.json \
        --peak_layers 20 21 22 23 \
        --output_dir results/phase2 \
        --n_pairs 256

    # Qwen3 (n_layers=36; use the peak layers found by activation_patching.py's
    # layer sweep for this specific checkpoint -- there is no pre-known
    # Qwen3 peak range yet, unlike llama/gemma):
    python component_analysis.py \
        --model_path <hf_path> \
        --model_name qwen3-sft \
        --arch qwen3 \
        --harmful_path refusal_direction/dataset/splits/harmful_train.json \
        --harmless_path refusal_direction/dataset/splits/harmless_train.json \
        --peak_layers 20 21 22 23 \
        --output_dir results/phase2 \
        --n_pairs 256
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from activation_patching import (
    ARCH_CONFIG,
    LLAMA_STRIP_TOKEN_IDS,
    get_last_meaningful_position,
    get_strip_token_ids,
    load_model,
    load_prompt_pairs,
    run_position_sanity_check,
    validate_pairs_generative,
)


def get_strip_ids(tokenizer, arch: str):
    """
    Delegates to activation_patching.get_strip_token_ids rather than
    re-implementing per-arch strip logic here. The previous local copy
    of this function only handled "llama" vs a bare else-branch that
    assumed Gemma's tokens -- any non-llama arch (including Qwen3)
    silently got the wrong strip set. Importing the single fixed
    implementation avoids this file and activation_patching.py
    drifting out of sync with each other again.
    """
    return get_strip_token_ids(tokenizer, arch)


def metric_from_outputs(outputs, pos: int, refusal_token_id: int, compliance_token_id: int):
    logits = outputs.logits[0, pos, :]
    return logits[refusal_token_id] - logits[compliance_token_id]


def _get_n_heads(model, arch: str) -> int:
    config_heads = getattr(model.config, "num_attention_heads", None)
    if config_heads is not None:
        return int(config_heads)
    if arch == "llama":
        return 32
    if arch == "gemma":
        return 16
    if arch == "qwen3":
        return 32  # verified: Qwen3-8B config, hidden_size 4096 / head_dim 128
    raise ValueError(f"Unknown arch: {arch}")


def attribution_patch_heads_at_layer(
    model,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    layer_idx: int,
    refusal_token_id: int,
    compliance_token_id: int,
    arch: str,
    strip_ids,
):
    """
    Return per-head attribution scores for one layer.

    The hook is on self_attn.o_proj input, which is the concatenated per-head
    attention result before the output projection. This is the closest stable
    HuggingFace hook point for Llama/Gemma without TransformerLens.
    """
    attn_module = model.model.layers[layer_idx].self_attn
    clean_pos = get_last_meaningful_position(clean_tokens, strip_ids)
    corrupted_pos = get_last_meaningful_position(corrupted_tokens, strip_ids)
    n_heads = _get_n_heads(model, arch)

    clean_store = {}

    def clean_hook(module, inputs, output):
        clean_store["act"] = inputs[0][0, clean_pos, :].detach().float().cpu()

    handle = attn_module.o_proj.register_forward_hook(clean_hook)
    try:
        with torch.no_grad():
            model(clean_tokens.unsqueeze(0).to(model.device))
    finally:
        handle.remove()

    corrupted_store = {}

    def corrupted_hook(module, inputs, output):
        act = inputs[0]
        act.retain_grad()
        corrupted_store["act"] = act

    handle = attn_module.o_proj.register_forward_hook(corrupted_hook)
    model.zero_grad(set_to_none=True)
    try:
        outputs = model(corrupted_tokens.unsqueeze(0).to(model.device))
        metric = metric_from_outputs(outputs, corrupted_pos, refusal_token_id, compliance_token_id)
        metric.backward()
    finally:
        handle.remove()

    corrupted_act = corrupted_store["act"]
    if corrupted_act.grad is None:
        raise RuntimeError("Attention o_proj input gradient is None")

    clean_vec = clean_store["act"].to(corrupted_act.device, dtype=corrupted_act.dtype)
    corrupt_vec = corrupted_act[0, corrupted_pos, :].detach()
    grad_vec = corrupted_act.grad[0, corrupted_pos, :]

    if clean_vec.numel() % n_heads != 0:
        raise RuntimeError(
            f"Cannot split attention activation of size {clean_vec.numel()} into {n_heads} heads"
        )

    head_dim = clean_vec.numel() // n_heads
    attrs = (grad_vec * (clean_vec - corrupt_vec)).reshape(n_heads, head_dim).sum(dim=1)
    return attrs.detach().float().cpu()


def attribution_patch_mlp_at_layer(
    model,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    layer_idx: int,
    refusal_token_id: int,
    compliance_token_id: int,
    strip_ids,
):
    """Return scalar attribution score for the MLP output at one layer."""
    mlp_module = model.model.layers[layer_idx].mlp
    clean_pos = get_last_meaningful_position(clean_tokens, strip_ids)
    corrupted_pos = get_last_meaningful_position(corrupted_tokens, strip_ids)

    clean_store = {}

    def clean_hook(module, inputs, output):
        clean_store["act"] = output[0, clean_pos, :].detach().float().cpu()

    handle = mlp_module.register_forward_hook(clean_hook)
    try:
        with torch.no_grad():
            model(clean_tokens.unsqueeze(0).to(model.device))
    finally:
        handle.remove()

    corrupted_store = {}

    def corrupted_hook(module, inputs, output):
        output.retain_grad()
        corrupted_store["act"] = output

    handle = mlp_module.register_forward_hook(corrupted_hook)
    model.zero_grad(set_to_none=True)
    try:
        outputs = model(corrupted_tokens.unsqueeze(0).to(model.device))
        metric = metric_from_outputs(outputs, corrupted_pos, refusal_token_id, compliance_token_id)
        metric.backward()
    finally:
        handle.remove()

    corrupted_act = corrupted_store["act"]
    if corrupted_act.grad is None:
        raise RuntimeError("MLP output gradient is None")

    clean_vec = clean_store["act"].to(corrupted_act.device, dtype=corrupted_act.dtype)
    corrupt_vec = corrupted_act[0, corrupted_pos, :].detach()
    grad_vec = corrupted_act.grad[0, corrupted_pos, :]
    return float((grad_vec * (clean_vec - corrupt_vec)).sum().detach().cpu())


def component_analysis(
    model,
    tokenizer,
    valid_pairs,
    refusal_token_ids,
    compliance_token_ids,
    peak_layers,
    arch: str,
    output_dir: Path,
    model_name: str,
):
    """
    For each peak layer, compute mean per-head attention attribution and mean
    MLP attribution across valid pairs. Save JSON and a .pt file with tensors.
    """
    results = {}
    tensor_results = {}
    strip_ids = get_strip_ids(tokenizer, arch)
    n_heads = _get_n_heads(model, arch)

    for layer_idx in peak_layers:
        print(f"\n  Analysing layer {layer_idx}...")
        head_attrs = []
        mlp_attrs = []

        iterator = zip(valid_pairs, refusal_token_ids, compliance_token_ids)
        for (clean_tokens, corrupted_tokens), refusal_id, compliance_id in tqdm(
            iterator,
            total=len(valid_pairs),
            desc=f"Layer {layer_idx}",
            leave=False,
        ):
            try:
                head_attr = attribution_patch_heads_at_layer(
                    model,
                    clean_tokens,
                    corrupted_tokens,
                    layer_idx,
                    refusal_id,
                    compliance_id,
                    arch,
                    strip_ids,
                )
                mlp_attr = attribution_patch_mlp_at_layer(
                    model,
                    clean_tokens,
                    corrupted_tokens,
                    layer_idx,
                    refusal_id,
                    compliance_id,
                    strip_ids,
                )
                head_attrs.append(head_attr)
                mlp_attrs.append(mlp_attr)
            except Exception as exc:
                print(f"    Warning: pair failed at layer {layer_idx}: {exc}")

        if head_attrs:
            head_tensor = torch.stack(head_attrs)
            mean_head_attrs = head_tensor.mean(dim=0)
            mean_attn = float(mean_head_attrs.sum().item())
            top_heads = torch.topk(mean_head_attrs.abs(), k=min(5, n_heads)).indices.tolist()
        else:
            head_tensor = torch.empty(0, n_heads)
            mean_head_attrs = torch.zeros(n_heads)
            mean_attn = 0.0
            top_heads = []

        mlp_array = np.array(mlp_attrs, dtype=np.float32)
        mean_mlp = float(mlp_array.mean()) if len(mlp_array) else 0.0

        layer_key = str(layer_idx)
        results[layer_key] = {
            "mean_attn_attribution": mean_attn,
            "mean_mlp_attribution": mean_mlp,
            "std_attn_total": float(head_tensor.sum(dim=1).std().item()) if len(head_attrs) > 1 else 0.0,
            "std_mlp": float(mlp_array.std()) if len(mlp_array) else 0.0,
            "mean_head_attributions": [float(x) for x in mean_head_attrs.tolist()],
            "top_abs_heads": [int(x) for x in top_heads],
            "n_pairs": len(head_attrs),
        }
        tensor_results[layer_key] = {
            "head_attributions": head_tensor,
            "mlp_attributions": torch.tensor(mlp_attrs, dtype=torch.float32),
        }
        print(f"    Attn total: {mean_attn:.4f}, MLP: {mean_mlp:.4f}, top heads: {top_heads}")

        del head_attrs, mlp_attrs
        torch.cuda.empty_cache()

    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    json_path = model_dir / "component_analysis.json"
    pt_path = model_dir / "component_analysis.pt"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    torch.save(tensor_results, pt_path)

    print(f"\nComponent analysis saved to {json_path}")
    print(f"Raw attribution tensors saved to {pt_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Component Analysis (Attribution Patching)")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--arch", required=True, choices=["llama", "gemma", "qwen3"])
    parser.add_argument("--harmful_path", required=True)
    parser.add_argument("--harmless_path", required=True)
    parser.add_argument(
        "--peak_layers",
        type=int,
        nargs="+",
        required=True,
        help="Layers to analyse, usually from layer_sweep_normalised top layers",
    )
    parser.add_argument("--output_dir", default="results/phase2")
    parser.add_argument("--n_pairs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_layer = ARCH_CONFIG[args.arch]["n_layers"] - 1
    bad_layers = [layer for layer in args.peak_layers if layer < 0 or layer > max_layer]
    if bad_layers:
        raise ValueError(f"Invalid peak layers for {args.arch}: {bad_layers}; valid range is 0..{max_layer}")

    tokenizer, model = load_model(args.model_path, args.arch)
    pairs = load_prompt_pairs(
        args.harmful_path,
        args.harmless_path,
        tokenizer,
        args.arch,
        model_name=args.model_name,
        n=args.n_pairs,
        seed=args.seed,
    )
    valid_pairs, refusal_token_ids, compliance_token_ids = validate_pairs_generative(
        model,
        tokenizer,
        pairs,
        model_name=args.model_name,
    )

    if len(valid_pairs) == 0:
        raise RuntimeError("No valid pairs from generative classification.")

    # Cheap pre-flight check before spending GPU time on attribution
    # patching -- catches position-mismatch bugs (like the strip-set
    # over-stripping issue found for qwen3/gemma) before, not after,
    # a full run.
    strip_ids = get_strip_token_ids(tokenizer, args.arch)
    run_position_sanity_check(tokenizer, valid_pairs, strip_ids)

    component_analysis(
        model,
        tokenizer,
        valid_pairs,
        refusal_token_ids,
        compliance_token_ids,
        args.peak_layers,
        args.arch,
        output_dir,
        args.model_name,
    )


if __name__ == "__main__":
    main()
