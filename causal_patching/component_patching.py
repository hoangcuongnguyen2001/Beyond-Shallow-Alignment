"""
Component-level activation patching: confirmatory full causal patching
of top attention heads and MLP outputs identified by component_analysis.py.

WHERE THIS SITS IN THE PIPELINE
--------------------------------
  activation_patching.py   → residual-stream layer sweep
  component_analysis.py    → attribution patching (gradient approximation)
  THIS SCRIPT              → confirmatory full causal patching

RA-SFT POSITION HANDLING (Llama / Gemma / Qwen3)
--------------------------------
Any Ra-SFT variant externalises its reasoning chain as <think>...</think>
before the actual refusal token. The refusal decision token is NOT at the
prompt boundary — it is at a position deep inside the generated sequence,
after </think>.

build_full_sequence() generates the full output, locates </think> by token-level
span matching, skips whitespace tokens, and returns the full token tensor
truncated at the decision token. get_measurement_pos() then returns the last
index of that tensor as the measurement position.

SEQUENCE CACHING
-----------------
build_full_sequence() is expensive for Ra-SFT (one generation per pair).
After the build loop completes, full sequences are cached to disk at:
  {output_dir}/{model_name}/sequence_cache.pt

On subsequent runs (e.g. after fixing position logic), the cache is loaded
instead of regenerating. Delete the cache file to force regeneration.
This protects against losing the generation cost if position debugging is needed.

KEY METRIC DIFFERENCE FROM THE LAYER SWEEP
-------------------------------------------
Layer sweep patches the FULL residual stream at layer L.
Component patching patches ONE component's output (MLP or single attention head).
Same normalised logit difference metric — enables direct comparison.
Component effects do NOT sum to the layer-level effect.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from activation_patching import (
    ARCH_CONFIG,
    get_strip_token_ids,
    get_last_meaningful_position,
    load_model,
    load_prompt_pairs,
    validate_pairs_generative,
    normalise_effect,
    run_position_sanity_check,
)


# ---------------------------------------------------------------------------
# Ra-SFT position helpers
# ---------------------------------------------------------------------------

def is_ra_sft(model_name: str) -> bool:
    return "ra-sft" in model_name.lower()


def build_full_sequence(
    model,
    tokenizer,
    tokens: torch.Tensor,
    model_name: str,
    arch: str,
    max_new_tokens: int = 512,
) -> torch.Tensor:
    """
    For Ra-SFT: generate the full sequence (prompt + reasoning chain +
    decision token) and return truncated at the decision token.

    For non-Ra-SFT: return tokens unchanged.

    FALLBACK BEHAVIOUR:
    If </think> is not found in the generated output (generation timed out,
    model produced no reasoning chain, or the prompt produced a very short
    output), returns the original prompt tokens unchanged. The pair will be
    flagged as invalid by valid_pos_mask in component_patching() and skipped.

    DEBUG: to verify the decision token is correct, after calling this function:
        pos = len(result) - 1
        print(tokenizer.decode([result[pos].item()]))
    This should print the first word of the refusal/compliance response,
    NOT a reasoning chain token or whitespace.
    """
    if not is_ra_sft(model_name):
        return tokens

    inp = tokens.unsqueeze(0).to(model.device)
    with torch.no_grad():
        out = model.generate(
            inp,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    full_tokens = out[0].cpu()
    prompt_len = tokens.shape[0]

    # Decode generated portion only (skip_special_tokens=False to preserve
    # </think> tag in the decoded string for reliable detection)
    generated = full_tokens[prompt_len:]
    decoded_generated = tokenizer.decode(generated.tolist(), skip_special_tokens=False)

    if "</think>" not in decoded_generated:
        # Reasoning chain did not complete — pair will be dropped
        return tokens

    # ---------------------------------------------------------------------------
    # Locate </think> at the TOKEN level in the full sequence.
    # We do NOT rely on the decoded string position because character offsets
    # do not map reliably to token positions for multi-byte tokens.
    #
    # Strategy: encode "</think>" to get its token ids, then find the span
    # in full_tokens. If the tokenizer splits </think> differently depending
    # on context (e.g. with a leading space), try both with and without
    # add_special_tokens to get the right ids.
    # ---------------------------------------------------------------------------
    full_list = full_tokens.tolist()

    # Try encoding </think> with and without leading newline context
    candidates = [
        tokenizer.encode("</think>", add_special_tokens=False),
        tokenizer.encode("\n</think>", add_special_tokens=False),
        tokenizer.encode(" </think>", add_special_tokens=False),
    ]
    # Deduplicate and sort by length descending (prefer longer matches)
    seen = set()
    think_candidates = []
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            think_candidates.append(c)
    think_candidates.sort(key=len, reverse=True)

    think_end_pos = None
    for think_ids in think_candidates:
        n = len(think_ids)
        for i in range(prompt_len, len(full_list) - n + 1):
            if full_list[i: i + n] == think_ids:
                think_end_pos = i + n
                break
        if think_end_pos is not None:
            break

    if think_end_pos is None:
        # Token-level search failed despite string-level </think> being present.
        # This can happen if </think> is tokenised as a single special token
        # not returned by encode(). Fall back to string-split approach:
        # find the character position of </think> in decoded_generated,
        # then encode the prefix up to that point and count tokens.
        split_point = decoded_generated.find("</think>") + len("</think>")
        prefix = decoded_generated[:split_point]
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        think_end_pos = prompt_len + len(prefix_ids)
        if think_end_pos >= len(full_list):
            return tokens

    # ---------------------------------------------------------------------------
    # Skip whitespace / newline / special tokens immediately after </think>.
    # These are formatting tokens, not the decision token.
    # We use a broader strip set here: Gemma end-of-turn tokens + newline tokens.
    # ---------------------------------------------------------------------------
    # Use the ACTUAL arch's strip tokens, not a hardcoded "gemma" --
    # previously this was always Gemma's <end_of_turn>/<start_of_turn>
    # tokens regardless of what model was actually running, which for
    # Llama or Qwen3 meant this set contained irrelevant token ids.
    # The newline_ids union below did most of the real work in practice,
    # which is why this went unnoticed, but it's worth being correct here
    # rather than relying on that overlap holding for every case.
    base_strip_ids = get_strip_token_ids(tokenizer, arch)

    # Also strip newline tokens (token id for "\n" varies by tokenizer)
    newline_ids = set(tokenizer.encode("\n", add_special_tokens=False))
    newline_ids.update(tokenizer.encode(" \n", add_special_tokens=False))
    newline_ids.update(tokenizer.encode("\n\n", add_special_tokens=False))
    extended_strip_ids = base_strip_ids | newline_ids

    pos = think_end_pos
    while pos < len(full_list) and full_list[pos] in extended_strip_ids:
        pos += 1

    if pos >= len(full_list):
        # Nothing after </think> — pair will be dropped
        return tokens

    # Verify: the token at pos should be a content token (refusal/compliance).
    # Print for debugging so position can be confirmed before full run.
    decision_token = tokenizer.decode([full_list[pos]], skip_special_tokens=False)
    print(f"    [Ra-SFT pos check] decision token at pos {pos}: '{decision_token}'")

    return full_tokens[: pos + 1]


def get_measurement_pos(
    full_tokens: torch.Tensor,
    prompt_tokens: torch.Tensor,
    model_name: str,
    strip_ids: set,
) -> int:
    """
    Return the position index in full_tokens at which to measure logits.

    SFT/ORPO: full_tokens == prompt_tokens; returns last non-strip token index.
    Ra-SFT:   full_tokens ends at the decision token; returns last index.
    """
    if not is_ra_sft(model_name):
        return get_last_meaningful_position(prompt_tokens, strip_ids)
    return len(full_tokens) - 1


# ---------------------------------------------------------------------------
# Arch helper
# ---------------------------------------------------------------------------

def get_n_heads(model, arch: str) -> int:
    n = getattr(model.config, "num_attention_heads", None)
    if n is not None:
        return int(n)
    return {"llama": 32, "gemma": 16, "qwen3": 32}[arch]


# ---------------------------------------------------------------------------
# Logit diff at a specific position
# ---------------------------------------------------------------------------

def compute_logit_diff_at_pos(
    model,
    tokens: torch.Tensor,
    pos: int,
    refusal_token_id: int,
    compliance_token_id: int,
) -> float:
    inp = tokens.unsqueeze(0).to(model.device)
    with torch.no_grad():
        outputs = model(inp)
    logits = outputs.logits[0, pos, :]
    return (logits[refusal_token_id] - logits[compliance_token_id]).item()


# ---------------------------------------------------------------------------
# Component patching functions
# ---------------------------------------------------------------------------

def patch_mlp_and_measure(
    model,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    layer_idx: int,
    refusal_token_id: int,
    compliance_token_id: int,
    clean_pos: int,
    corrupted_pos: int,
) -> float:
    """
    Patch the CLEAN MLP output at layer_idx into the CORRUPTED run.

    Sign interpretation:
      Positive: MLP causally promotes refusal
      Negative: MLP suppresses refusal in the harmful run
      Near zero: MLP output difference is not load-bearing
    """
    mlp_module = model.model.layers[layer_idx].mlp
    clean_store = {}

    def clean_hook(module, inputs, output):
        clean_store["act"] = output[0, clean_pos, :].detach().float().cpu()

    handle = mlp_module.register_forward_hook(clean_hook)
    try:
        with torch.no_grad():
            model(clean_tokens.unsqueeze(0).to(model.device))
    finally:
        handle.remove()

    clean_mlp = clean_store["act"].to(model.device)
    patched = [False]

    def patch_hook(module, inputs, output):
        if patched[0]:
            return output
        out = output.clone()
        out[0, corrupted_pos, :] = clean_mlp.to(output.dtype)
        patched[0] = True
        return out

    handle = mlp_module.register_forward_hook(patch_hook)
    try:
        with torch.no_grad():
            outputs = model(corrupted_tokens.unsqueeze(0).to(model.device))
    finally:
        handle.remove()

    logits = outputs.logits[0, corrupted_pos, :]
    return (logits[refusal_token_id] - logits[compliance_token_id]).item()


def patch_attention_head_and_measure(
    model,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    layer_idx: int,
    head_idx: int,
    refusal_token_id: int,
    compliance_token_id: int,
    arch: str,
    clean_pos: int,
    corrupted_pos: int,
) -> float:
    """
    Patch ONE clean attention head output into the corrupted run at layer_idx.

    Hook point: INPUT to o_proj. Only head_idx slice is replaced.
    Downstream effect: W_o[:, H*hd:(H+1)*hd] @ (clean_head_H - corrupted_head_H).

    Sign interpretation:
      Positive: this head carries refusal signal — patching restores refusal.
      Negative: this head is suppressive in the harmful run.
      Large |attribution_approx| but small |causal_effect|: gradient
        approximation overestimated importance due to nonlinearity or
        cancellation downstream.
    """
    attn_module = model.model.layers[layer_idx].self_attn
    n_heads = get_n_heads(model, arch)
    clean_store = {}

    def clean_hook(module, inputs, output):
        act = inputs[0][0, clean_pos, :]
        head_dim = act.numel() // n_heads
        clean_store["head"] = act[
            head_idx * head_dim: (head_idx + 1) * head_dim
        ].detach().float().cpu()

    handle = attn_module.o_proj.register_forward_hook(clean_hook)
    try:
        with torch.no_grad():
            model(clean_tokens.unsqueeze(0).to(model.device))
    finally:
        handle.remove()

    clean_head = clean_store["head"].to(model.device)
    patched = [False]

    def patch_hook(module, inputs, output):
        if patched[0]:
            return output
        act = inputs[0].clone()
        head_dim = act.shape[-1] // n_heads
        act[0, corrupted_pos,
            head_idx * head_dim: (head_idx + 1) * head_dim
            ] = clean_head.to(act.dtype)
        patched[0] = True
        return module(act)

    handle = attn_module.o_proj.register_forward_hook(patch_hook)
    try:
        with torch.no_grad():
            outputs = model(corrupted_tokens.unsqueeze(0).to(model.device))
    finally:
        handle.remove()

    logits = outputs.logits[0, corrupted_pos, :]
    return (logits[refusal_token_id] - logits[compliance_token_id]).item()


# ---------------------------------------------------------------------------
# Main analysis loop
# ---------------------------------------------------------------------------

def component_patching(
    model,
    tokenizer,
    valid_pairs,
    refusal_token_ids,
    compliance_token_ids,
    component_json: dict,
    arch: str,
    model_name: str,
    output_dir: Path,
    top_k: int = 5,
):
    strip_ids = get_strip_token_ids(tokenizer, arch)
    ra_sft = is_ra_sft(model_name)

    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    cache_path = model_dir / "sequence_cache.pt"

    # ------------------------------------------------------------------
    # Pre-compute full sequences — load from cache if available.
    # Cache is written after the build loop so a failed/interrupted run
    # does not produce a partial cache file.
    # ------------------------------------------------------------------
    print(
        f"\nPre-computing measurement positions "
        f"({'full generation required for Ra-SFT' if ra_sft else 'prompt boundary'})..."
    )

    if cache_path.exists():
        print(f"  Loading sequence cache from {cache_path}")
        cache = torch.load(cache_path)
        full_clean_list = cache["full_clean"]
        full_corrupted_list = cache["full_corrupted"]
        print(f"  Loaded {len(full_clean_list)} cached sequences.")
    else:
        full_clean_list = []
        full_corrupted_list = []

        for (clean_tokens, corrupted_tokens) in tqdm(
            valid_pairs, desc="Building sequences"
        ):
            full_clean = build_full_sequence(
                model, tokenizer, clean_tokens, model_name, arch
            )
            full_corrupted = build_full_sequence(
                model, tokenizer, corrupted_tokens, model_name, arch
            )
            full_clean_list.append(full_clean)
            full_corrupted_list.append(full_corrupted)

        # Save cache immediately after build loop completes.
        # If the build loop was interrupted, cache_path will not exist and
        # the next run will regenerate from scratch.
        torch.save(
            {"full_clean": full_clean_list, "full_corrupted": full_corrupted_list},
            cache_path,
        )
        print(f"  Sequence cache saved to {cache_path}")

    # ------------------------------------------------------------------
    # Compute measurement positions from full sequences
    # ------------------------------------------------------------------
    clean_pos_list = []
    corrupted_pos_list = []
    valid_pos_mask = []

    for i, (clean_tokens, corrupted_tokens) in enumerate(valid_pairs):
        full_clean = full_clean_list[i]
        full_corrupted = full_corrupted_list[i]
        c_pos = get_measurement_pos(full_clean, clean_tokens, model_name, strip_ids)
        p_pos = get_measurement_pos(
            full_corrupted, corrupted_tokens, model_name, strip_ids
        )
        clean_pos_list.append(c_pos)
        corrupted_pos_list.append(p_pos)
        valid_pos_mask.append(c_pos > 0 and p_pos > 0)

    n_valid = sum(valid_pos_mask)
    print(f"  Valid positions: {n_valid}/{len(valid_pairs)}")
    if ra_sft and n_valid < len(valid_pairs):
        print(
            f"  NOTE: {len(valid_pairs) - n_valid} Ra-SFT pairs dropped — "
            f"</think> detection failed or position was degenerate."
        )

    results = {}
    tensor_results = {}

    for layer_str, layer_data in component_json.items():
        layer_idx = int(layer_str)
        print(f"\n=== Layer {layer_idx} ===")

        head_attrs = layer_data["mean_head_attributions"]
        top_heads = sorted(
            range(len(head_attrs)),
            key=lambda h: abs(head_attrs[h]),
            reverse=True,
        )[:top_k]

        print(f"  Top-{top_k} heads by |attribution|: {top_heads}")
        print(
            f"  MLP attribution (from gradient approx): "
            f"{layer_data['mean_mlp_attribution']:.4f}"
        )

        layer_results = {}
        layer_tensors = {}

        # ---- MLP patching ------------------------------------------------
        print(f"  Patching MLP at layer {layer_idx}...")
        mlp_norm_effects = []

        for i, ((clean_tokens, corrupted_tokens), refusal_id, compliance_id) in enumerate(
            tqdm(
                zip(valid_pairs, refusal_token_ids, compliance_token_ids),
                total=len(valid_pairs),
                desc=f"  L{layer_idx} MLP",
                leave=False,
            )
        ):
            if not valid_pos_mask[i]:
                continue

            full_clean = full_clean_list[i]
            full_corrupted = full_corrupted_list[i]
            c_pos = clean_pos_list[i]
            p_pos = corrupted_pos_list[i]

            try:
                c_metric = compute_logit_diff_at_pos(
                    model, full_clean, c_pos, refusal_id, compliance_id
                )
                h_metric = compute_logit_diff_at_pos(
                    model, full_corrupted, p_pos, refusal_id, compliance_id
                )
                denom = c_metric - h_metric
                if abs(denom) < 0.5:
                    continue

                patched_metric = patch_mlp_and_measure(
                    model, full_clean, full_corrupted,
                    layer_idx, refusal_id, compliance_id,
                    clean_pos=c_pos, corrupted_pos=p_pos,
                )
                norm = (patched_metric - h_metric) / denom
                mlp_norm_effects.append(norm)
            except Exception as exc:
                print(f"    Warning: MLP patch failed at pair {i}: {exc}")

        mean_mlp = float(np.mean(mlp_norm_effects)) if mlp_norm_effects else 0.0
        std_mlp = (
            float(np.std(mlp_norm_effects)) if len(mlp_norm_effects) > 1 else 0.0
        )
        print(f"    MLP normalised effect: {mean_mlp:.4f} ± {std_mlp:.4f}")

        layer_results["mlp"] = {
            "mean_normalised_effect": mean_mlp,
            "std_normalised_effect": std_mlp,
            "attribution_approx": layer_data["mean_mlp_attribution"],
            "n_pairs": len(mlp_norm_effects),
        }
        layer_tensors["mlp_norm_effects"] = torch.tensor(
            mlp_norm_effects, dtype=torch.float32
        )

        # ---- Attention head patching -------------------------------------
        head_results = {}
        head_tensors = {}

        for head_idx in top_heads:
            print(f"  Patching head {head_idx} at layer {layer_idx}...")
            head_norm_effects = []

            for i, (
                (clean_tokens, corrupted_tokens), refusal_id, compliance_id
            ) in enumerate(
                tqdm(
                    zip(valid_pairs, refusal_token_ids, compliance_token_ids),
                    total=len(valid_pairs),
                    desc=f"  L{layer_idx} H{head_idx}",
                    leave=False,
                )
            ):
                if not valid_pos_mask[i]:
                    continue

                full_clean = full_clean_list[i]
                full_corrupted = full_corrupted_list[i]
                c_pos = clean_pos_list[i]
                p_pos = corrupted_pos_list[i]

                try:
                    c_metric = compute_logit_diff_at_pos(
                        model, full_clean, c_pos, refusal_id, compliance_id
                    )
                    h_metric = compute_logit_diff_at_pos(
                        model, full_corrupted, p_pos, refusal_id, compliance_id
                    )
                    denom = c_metric - h_metric
                    if abs(denom) < 0.5:
                        continue

                    patched_metric = patch_attention_head_and_measure(
                        model, full_clean, full_corrupted,
                        layer_idx, head_idx,
                        refusal_id, compliance_id,
                        arch,
                        clean_pos=c_pos, corrupted_pos=p_pos,
                    )
                    norm = (patched_metric - h_metric) / denom
                    head_norm_effects.append(norm)
                except Exception as exc:
                    print(
                        f"    Warning: head patch failed at pair {i}: {exc}"
                    )

            mean_head = (
                float(np.mean(head_norm_effects)) if head_norm_effects else 0.0
            )
            std_head = (
                float(np.std(head_norm_effects))
                if len(head_norm_effects) > 1
                else 0.0
            )
            print(
                f"    Head {head_idx} normalised effect: "
                f"{mean_head:.4f} ± {std_head:.4f}"
                f"  (attribution approx was {head_attrs[head_idx]:.4f})"
            )

            head_results[str(head_idx)] = {
                "mean_normalised_effect": mean_head,
                "std_normalised_effect": std_head,
                "attribution_approx": head_attrs[head_idx],
                "n_pairs": len(head_norm_effects),
            }
            head_tensors[f"head_{head_idx}_norm_effects"] = torch.tensor(
                head_norm_effects, dtype=torch.float32
            )

        layer_results["attention_heads"] = head_results
        layer_tensors["head_tensors"] = head_tensors
        results[layer_str] = layer_results
        tensor_results[layer_str] = layer_tensors

    # ---- Save ------------------------------------------------------------
    json_path = model_dir / "component_patching.json"
    pt_path = model_dir / "component_patching.pt"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    torch.save(tensor_results, pt_path)

    print(f"\nComponent patching results saved to {json_path}")
    print(f"Raw effect tensors saved to {pt_path}")
    return results


def print_summary(results: dict):
    print("\n" + "=" * 70)
    print("SUMMARY: Attribution approx vs confirmed causal effect")
    print("=" * 70)
    print(
        f"{'Layer':>6}  {'Component':>12}  "
        f"{'Attrib approx':>14}  {'Causal effect':>14}  {'n':>5}"
    )
    print("-" * 70)

    rows = []
    for layer_str, layer_data in results.items():
        m = layer_data["mlp"]
        rows.append((
            layer_str, "MLP",
            m["attribution_approx"], m["mean_normalised_effect"], m["n_pairs"],
        ))
        for head_str, h in layer_data["attention_heads"].items():
            rows.append((
                layer_str, f"Head {head_str}",
                h["attribution_approx"], h["mean_normalised_effect"], h["n_pairs"],
            ))

    rows.sort(key=lambda r: abs(r[3]), reverse=True)
    for layer_str, comp, approx, causal, n in rows:
        print(
            f"{layer_str:>6}  {comp:>12}  {approx:>+14.4f}  "
            f"{causal:>+14.4f}  {n:>5}"
        )
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Component-level activation patching (confirmatory, full causal)"
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--model_name", required=True,
        help="e.g. gemma-sft, gemma-ra-sft, llama-orpo, qwen3-ra-sft",
    )
    parser.add_argument("--arch", required=True, choices=["llama", "gemma", "qwen3"])
    parser.add_argument("--harmful_path", required=True)
    parser.add_argument("--harmless_path", required=True)
    parser.add_argument("--component_json", required=True)
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Number of top heads per layer (default 5)",
    )
    parser.add_argument("--output_dir", default="results/phase2")
    parser.add_argument("--n_pairs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.component_json) as f:
        component_json = json.load(f)

    tokenizer, model = load_model(args.model_path, args.arch)
    pairs = load_prompt_pairs(
        args.harmful_path, args.harmless_path,
        tokenizer, args.arch,
        model_name=args.model_name,
        n=args.n_pairs, seed=args.seed,
    )
    valid_pairs, refusal_token_ids, compliance_token_ids = validate_pairs_generative(
        model, tokenizer, pairs, model_name=args.model_name
    )

    if len(valid_pairs) == 0:
        raise RuntimeError("No valid pairs from generative classification.")

    # Cheap pre-flight check on the raw prompt boundary before the
    # expensive sequence-building loop (which, for Ra-SFT, means one
    # full generation per pair). Checks the SAME strip-set logic that
    # both build_full_sequence's post-</think> skip and get_measurement_pos's
    # SFT/ORPO path depend on -- catches a mis-defined strip set before
    # burning GPU time on generations built from the wrong position.
    strip_ids = get_strip_token_ids(tokenizer, args.arch)
    run_position_sanity_check(tokenizer, valid_pairs, strip_ids)

    results = component_patching(
        model, tokenizer,
        valid_pairs, refusal_token_ids, compliance_token_ids,
        component_json=component_json,
        arch=args.arch,
        model_name=args.model_name,
        output_dir=output_dir,
        top_k=args.top_k,
    )

    print_summary(results)


if __name__ == "__main__":
    main()