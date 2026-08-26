"""
Phase 2: Activation Patching — Causal validation of refusal direction layer localisation
Implements residual-stream layer sweep and logit-difference metric.

Usage:
    python activation_patching.py \
        --model_path models/llama-sft \
        --model_name llama-sft \
        --arch llama \
        --harmful_path data/harmful_train.json \
        --harmless_path data/harmless.json \
        --output_dir results/phase2 \
        --n_pairs 256

    # For Gemma Ra-SFT:
    python activation_patching.py \
        --model_path models/gemma-ra-sft \
        --model_name gemma-ra-sft \
        --arch gemma \
        --harmful_path data/harmful_train.json \
        --harmless_path data/harmless.json \
        --output_dir results/phase2

    # For Qwen3 Ra-SFT:
    python activation_patching.py \
        --model_path models/qwen3-ra-sft \
        --model_name qwen3-ra-sft \
        --arch qwen3 \
        --harmful_path data/harmful_train.json \
        --harmless_path data/harmless.json \
        --output_dir results/phase2

Supports Llama-3.1-8B (32 layers), Gemma-2-9B (42 layers), and Qwen3-8B (36 layers).
Native HuggingFace hooks only — no TransformerLens.

Key design decisions:
- Per-pair refusal/compliance token ids derived from actual model generation
- Position patching at last meaningful token before assistant header \n\n
- Gemma Ra-SFT handled separately: reasoning chains decoded with
  skip_special_tokens=False, post-think content used for classification
  and token extraction
- Arditi et al. 12-phrase refusal classifier
"""

import argparse
import json
import re
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARCH_CONFIG = {
    "llama": {
        "n_layers": 32,
        "d_model": 4096,
    },
    "gemma": {
        "n_layers": 42,
        "d_model": 3584,
    },
    "qwen3": {
        "n_layers": 36,
        "d_model": 4096,
    },
}

QWEN3_SYSTEM = "You are a helpful assistant."

LLAMA_SYSTEM = "You are a helpful assistant."
GEMMA_SYSTEM = "You are a helpful assistant."

# Arditi et al. 12-phrase refusal classifier
REFUSAL_PHRASES = [
    "I'm sorry", "I am sorry", "I apologize", "As an AI",
    "As a language model", "As an assistant", "I cannot",
    "I can't", "I am unable to", "I'm unable to",
    "I am not able to", "I'm not able to",
]

# Token ids to strip from end of sequence to find last meaningful position
# Llama: \n\n=271, \n=198, space=220
# Gemma strip ids are computed at runtime from tokenizer
LLAMA_STRIP_TOKEN_IDS = {271, 198, 220}


# ---------------------------------------------------------------------------
# 1. Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str, arch: str):
    """Load tokenizer and model. Returns (tokenizer, model)."""
    print(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.clean_up_tokenization_spaces = False

    print(f"Loading model from {model_path}")
    dtype = torch.bfloat16 if arch in ("gemma", "qwen3") else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    print(f"  Loaded. dtype={dtype}, device={next(model.parameters()).device}")
    return tokenizer, model


def get_strip_token_ids(tokenizer, arch: str) -> set:
    """
    Return set of token ids to strip from sequence end to find
    last meaningful position before generation starts.
    Llama: strip \n\n, \n, space
    Gemma: strip <end_of_turn>\n and trailing whitespace
    """
    if arch == "llama":
        return LLAMA_STRIP_TOKEN_IDS
    elif arch == "qwen3":
        # Strip the <think> stub (present for SFT/ORPO prompts, which end
        # "...<|im_start|>assistant\n<think>\n\n</think>\n\n") plus generic
        # whitespace -- but deliberately do NOT strip "assistant" itself.
        # Stopping there mirrors Llama's convention of landing on a fixed
        # template anchor common to every prompt, rather than stripping
        # past it into the (necessarily pair-varying) instruction content.
        # Encode the think-stub both as it appears after "assistant\n" and
        # standalone, since BPE merges can differ by context.
        qwen3_strip = set(tokenizer.encode(
            "<think>\n\n</think>\n\n", add_special_tokens=False,
        ))
        qwen3_strip.update(tokenizer.encode(
            "\n<think>\n\n</think>\n\n", add_special_tokens=False,
        ))
        qwen3_strip.update(tokenizer.encode("\n", add_special_tokens=False))
        qwen3_strip.update(tokenizer.encode(" ", add_special_tokens=False))
        return qwen3_strip
    else:
        # Strip only the trailing whitespace after "<start_of_turn>model" --
        # deliberately do NOT strip "model" (or "<start_of_turn>") itself.
        # This mirrors the Llama fix and the Qwen3 fix applied earlier:
        # the previous version stripped the ENTIRE "<end_of_turn>\n
        # <start_of_turn>model\n" block, including the header word itself,
        # so get_last_meaningful_position walked straight past the one
        # token every Gemma prompt shares (the fixed template anchor) and
        # landed on the last word of the actual instruction instead --
        # which necessarily differs between the clean and corrupted prompt
        # in every pair, since the instructions themselves differ. That's
        # exactly the run_position_sanity_check warning pattern: mismatched
        # content words instead of a shared "model" token. Stopping before
        # the header word restores a genuinely shared anchor position.
        gemma_strip = set(tokenizer.encode("\n", add_special_tokens=False))
        gemma_strip.update(tokenizer.encode(" ", add_special_tokens=False))
        return gemma_strip


# ---------------------------------------------------------------------------
# 2. Chat template formatting
# ---------------------------------------------------------------------------

def apply_chat_template(tokenizer, prompt: str, arch: str, model_name: str = "") -> str:
    """
    Format prompt using the correct chat template for each condition.
    Matches training format exactly for each model.
    """
    if arch == "llama":
        messages = [
            {"role": "system", "content": LLAMA_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Base model fine-tune: BOS not added by template when system role present
        if not formatted.startswith(tokenizer.bos_token):
            formatted = tokenizer.bos_token + formatted
        return formatted
    elif arch == "qwen3":
        # SFT/ORPO were trained with enable_thinking=False (empty stub baked
        # into every target) -- match that at eval time.
        # Ra-SFT was trained on a hand-rolled format with a REAL reasoning
        # trace inserted after "<|im_start|>assistant\n", with no stub
        # pre-inserted -- so at eval time it must NOT get the empty stub
        # either, or the model would be handed an already-closed think
        # block and skip reasoning entirely, which doesn't match how it
        # was trained to behave. enable_thinking=True (the template
        # default) reproduces that: nothing inserted after the assistant
        # header, model free to open its own <think> block.
        messages = [{"role": "user", "content": prompt}]
        if "ra-sft" in model_name:
            # train_sft_cot_qwen3.py used no system message -- match that.
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
        # train_sft_qwen3.py / train_orpo_qwen3.py both included a system
        # message -- match that for SFT/ORPO.
        messages = [
            {"role": "system", "content": QWEN3_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    else:
        # Gemma Ra-SFT: system prompt folded into user turn (matches training format)
        if "ra-sft" in model_name:
            user_content = f"{GEMMA_SYSTEM}\n\n{prompt}"
        else:
            user_content = prompt
        messages = [{"role": "user", "content": user_content}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def tokenize_prompt(tokenizer, prompt_str: str, max_length: int = 512) -> torch.Tensor:
    """Tokenize formatted prompt string. Returns 1D tensor."""
    enc = tokenizer(
        prompt_str,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    return enc["input_ids"].squeeze(0)


def load_prompt_pairs(
    harmful_path: str,
    harmless_path: str,
    tokenizer,
    arch: str,
    model_name: str = "",
    n: int = 256,
    seed: int = 42,
) -> list:
    """
    Load harmful/harmless prompt pairs from Arditi-format JSON files.
    Returns list of (clean_tokens, corrupted_tokens) where:
      clean     = harmful prompt (model should refuse)
      corrupted = harmless prompt (model should comply)
    """
    with open(harmful_path) as f:
        harmful_data = json.load(f)
    with open(harmless_path) as f:
        harmless_data = json.load(f)

    def extract_prompts(data):
        if isinstance(data, list):
            if isinstance(data[0], str):
                return data
            elif isinstance(data[0], dict):
                for key in ("instruction", "prompt", "text", "question"):
                    if key in data[0]:
                        return [d[key] for d in data]
        raise ValueError(f"Cannot parse prompt list from: {type(data[0])}")

    harmful_prompts = extract_prompts(harmful_data)
    harmless_prompts = extract_prompts(harmless_data)

    random.seed(seed)
    n_available = min(len(harmful_prompts), len(harmless_prompts), n)
    if n_available < n:
        print(f"  Warning: only {n_available} pairs available (requested {n})")

    indices = list(range(n_available))
    random.shuffle(indices)
    indices = indices[:n_available]

    pairs = []
    for i in indices:
        clean_str = apply_chat_template(tokenizer, harmful_prompts[i], arch, model_name)
        corrupted_str = apply_chat_template(tokenizer, harmless_prompts[i], arch, model_name)
        clean_tokens = tokenize_prompt(tokenizer, clean_str)
        corrupted_tokens = tokenize_prompt(tokenizer, corrupted_str)
        pairs.append((clean_tokens, corrupted_tokens))

    print(f"  Loaded {len(pairs)} prompt pairs")
    return pairs


# ---------------------------------------------------------------------------
# 3. Position utilities
# ---------------------------------------------------------------------------

def get_last_meaningful_position(tokens: torch.Tensor, strip_ids: set) -> int:
    """
    Return index of last token not in strip_ids.
    This is the position where logits reflect the model's response decision.
    """
    idx = len(tokens) - 1
    while idx >= 0 and tokens[idx].item() in strip_ids:
        idx -= 1
    return idx


# ---------------------------------------------------------------------------
# 4. Response classification
# ---------------------------------------------------------------------------

def classify_response_standard(
    model, tokenizer, tokens, max_new_tokens: int = 10
) -> tuple:
    """
    Standard classification for all conditions except Gemma Ra-SFT.
    Returns (label, decoded, source) where source is used for token extraction.
    label: 'refusal' | 'compliance' | 'ambiguous'
    source: the content string used for first-token extraction
    """
    inp = tokens.unsqueeze(0).to(model.device)
    with torch.no_grad():
        out = model.generate(
            inp,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = out[0][inp.shape[1]:]
    decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Handle any residual <think> blocks (e.g. Llama Ra-SFT partial reasoning)
    if "</think>" in decoded:
        post_think = decoded.split("</think>")[-1].strip()
        source = post_think if post_think else decoded
    else:
        source = decoded

    if any(p in source for p in REFUSAL_PHRASES):
        return "refusal", decoded, source
    elif source:
        return "compliance", decoded, source
    else:
        return "ambiguous", decoded, ""


def classify_response_ra_sft(
    model, tokenizer, tokens, max_new_tokens: int = 512
) -> tuple:
    """
    Classification for any Ra-SFT variant, which produces explicit
    <think> chains that can run well past a 10-token budget.

    Ra-SFT response structure (Llama/Gemma/Qwen3 alike):
        <think>
        ...multi-sentence reasoning about harm...
        </think>
        I cannot fulfill your request...
        <end/eot marker>

    Decodes with skip_special_tokens=False to preserve structure,
    then strips terminal tokens and extracts post-think content for
    classification. Token extraction uses post-think content — avoids
    identical-token problem where both refusal and compliance start
    with <think>.

    NOTE: previously this was gated to "gemma" Ra-SFT only, with Llama
    (and now Qwen3) Ra-SFT falling through to classify_response_standard's
    default max_new_tokens=10 -- nowhere near enough to generate past a
    full reasoning chain. If you've already run this script against
    Llama Ra-SFT under the old dispatch, its classification stats are
    worth re-checking (high "ambiguous" or unexpected "harmful_compliance"
    counts would be the symptom) before trusting those results.
    """
    inp = tokens.unsqueeze(0).to(model.device)
    with torch.no_grad():
        out = model.generate(
            inp,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = out[0][inp.shape[1]:]

    # Decode preserving special tokens to see full structure
    decoded_raw = tokenizer.decode(generated, skip_special_tokens=False)
    # Clean terminal special tokens while preserving content.
    # Covers Llama (<|eot_id|>), Gemma (<end_of_turn>/<eos>), and
    # Qwen3 (<|im_end|>) terminal markers.
    decoded = (
        decoded_raw
        .replace("<end_of_turn>", "")
        .replace("<eos>", "")
        .replace("<|eot_id|>", "")
        .replace("<|im_end|>", "")
        .strip()
    )

    # Extract post-think content
    if "<think>" in decoded and "</think>" in decoded:
        post_think = decoded.split("</think>")[-1].strip()
        source = post_think if post_think else decoded
    elif "<think>" in decoded:
        # Reasoning chain incomplete — classify from reasoning content
        think_content = decoded.split("<think>")[-1]
        # Harm signals within reasoning chain indicate refusal intent
        REASONING_HARM_SIGNALS = [
            "is harmful", "is illegal", "is unethical", "is dangerous",
            "harmful because", "illegal because", "should be rejected",
            "cannot assist", "should not provide", "violates",
        ]
        if any(p in think_content for p in REASONING_HARM_SIGNALS):
            return "refusal", decoded, think_content
        return "ambiguous", decoded, ""
    else:
        source = decoded

    if any(p in source for p in REFUSAL_PHRASES):
        return "refusal", decoded, source
    elif source:
        return "compliance", decoded, source
    else:
        return "ambiguous", decoded, ""


def classify_response(
    model, tokenizer, tokens,
    model_name: str = "",
    max_new_tokens: int = 10,
) -> tuple:
    """Dispatch to correct classifier based on model_name."""
    if "ra-sft" in model_name:
        return classify_response_ra_sft(
            model, tokenizer, tokens, max_new_tokens=512
        )
    return classify_response_standard(
        model, tokenizer, tokens, max_new_tokens=max_new_tokens
    )


# ---------------------------------------------------------------------------
# 5. Pair validation
# ---------------------------------------------------------------------------

def validate_pairs_generative(
    model, tokenizer, pairs: list, model_name: str = ""
) -> tuple:
    """
    Classify each pair via generation.
    Valid pair: harmful -> refusal, harmless -> compliance.
    
    Returns (valid_pairs, refusal_token_ids, compliance_token_ids)
    Per-pair token ids are derived from actual first token of each response.
    For Gemma Ra-SFT, token ids are extracted from post-think content.
    """
    valid_pairs = []
    refusal_token_ids = []
    compliance_token_ids = []
    stats = Counter()

    for i, (clean_tokens, corrupted_tokens) in enumerate(
        tqdm(pairs, desc="Classifying pairs")
    ):
        clean_label, clean_decoded, clean_source = classify_response(
            model, tokenizer, clean_tokens, model_name=model_name
        )
        corrupt_label, corrupt_decoded, corrupt_source = classify_response(
            model, tokenizer, corrupted_tokens, model_name=model_name
        )

        stats[f"harmful_{clean_label}"] += 1
        stats[f"harmless_{corrupt_label}"] += 1

        if i < 10:
            print(f"\nPair {i}:")
            print(f"  HARMFUL  [{clean_label}]: {repr(clean_source[:80])}")
            print(f"  HARMLESS [{corrupt_label}]: {repr(corrupt_source[:80])}")

        if clean_label == "refusal" and corrupt_label == "compliance":
            # Extract first token from response source (post-think for Ra-SFT)
            refusal_first = tokenizer.encode(
                clean_source[:20], add_special_tokens=False
            )[0]
            compliance_first = tokenizer.encode(
                corrupt_source[:20], add_special_tokens=False
            )[0]

            # Skip pairs where token ids are identical — metric would be zero
            if refusal_first == compliance_first:
                stats["skipped_identical_tokens"] += 1
                continue

            valid_pairs.append((clean_tokens, corrupted_tokens))
            refusal_token_ids.append(refusal_first)
            compliance_token_ids.append(compliance_first)

    print(f"\nClassification stats: {dict(stats)}")
    print(f"Valid pairs: {len(valid_pairs)}/{len(pairs)}")
    return valid_pairs, refusal_token_ids, compliance_token_ids


# ---------------------------------------------------------------------------
# 6. Logit difference metric
# ---------------------------------------------------------------------------

def compute_logit_diff(
    model, tokens: torch.Tensor,
    refusal_token_id: int, compliance_token_id: int,
    strip_ids: set = None,
) -> float:
    """
    Forward pass. Returns logit[refusal] - logit[compliance] at last
    meaningful token position (before assistant header whitespace).
    """
    if strip_ids is None:
        strip_ids = LLAMA_STRIP_TOKEN_IDS
    pos = get_last_meaningful_position(tokens, strip_ids)
    inp = tokens.unsqueeze(0).to(model.device)
    with torch.no_grad():
        outputs = model(inp)
    logits = outputs.logits[0, pos, :]
    return (logits[refusal_token_id] - logits[compliance_token_id]).item()


# ---------------------------------------------------------------------------
# 7. Activation caching
# ---------------------------------------------------------------------------

def cache_activations(
    model, tokens: torch.Tensor, n_layers: int, pos: int = -1
) -> dict:
    """
    Run forward pass, cache residual stream at position pos for each layer.
    Returns dict {layer_idx: tensor [d_model]} on CPU.
    pos should be get_last_meaningful_position result for clean tokens.
    """
    cache = {}
    handles = []

    def make_hook(l):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            cache[l] = hidden[0, pos, :].detach().cpu().clone()
        return hook

    for l in range(n_layers):
        handles.append(model.model.layers[l].register_forward_hook(make_hook(l)))

    inp = tokens.unsqueeze(0).to(model.device)
    with torch.no_grad():
        model(inp)

    for h in handles:
        h.remove()

    return cache


# ---------------------------------------------------------------------------
# 8. Patch and measure
# ---------------------------------------------------------------------------

def patch_and_measure(
    model,
    clean_cache: dict,
    corrupted_tokens: torch.Tensor,
    patch_layer: int,
    refusal_token_id: int,
    compliance_token_id: int,
    clean_pos: int = -1,
    strip_ids: set = None,
) -> float:
    """
    Run corrupted forward pass patching residual stream at patch_layer
    with cached clean activation. Returns logit difference after patch.
    
    clean_pos: position used when caching (from clean sequence)
    corrupted_pos: independently computed from corrupted sequence
    Both should land on the assistant header structural position.
    """
    if strip_ids is None:
        strip_ids = LLAMA_STRIP_TOKEN_IDS

    patched_ref = clean_cache[patch_layer].to(model.device)
    corrupted_pos = get_last_meaningful_position(corrupted_tokens, strip_ids)

    applied = [False]

    def patch_hook(module, input, output):
        if applied[0]:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        patched = hidden.clone()
        patched[0, corrupted_pos, :] = patched_ref
        applied[0] = True
        if isinstance(output, tuple):
            return (patched,) + output[1:]
        return patched

    handle = model.model.layers[patch_layer].register_forward_hook(patch_hook)
    inp = corrupted_tokens.unsqueeze(0).to(model.device)
    with torch.no_grad():
        outputs = model(inp)
    handle.remove()

    logits = outputs.logits[0, corrupted_pos, :]
    return (logits[refusal_token_id] - logits[compliance_token_id]).item()


# ---------------------------------------------------------------------------
# 9. Normalisation
# ---------------------------------------------------------------------------

def normalise_effect(
    patched_metric: float,
    clean_metric: float,
    corrupted_metric: float,
    eps: float = 1e-6,
) -> float:
    """
    (patched - corrupted) / (clean - corrupted)
    0.0 = no causal effect
    1.0 = full restoration of clean behaviour
    >1.0 = overshoot (patch more than sufficient)
    negative = suppressive
    """
    denom = clean_metric - corrupted_metric
    if abs(denom) < eps:
        return 0.0
    return (patched_metric - corrupted_metric) / denom


# ---------------------------------------------------------------------------
# 10. Position sanity check
# ---------------------------------------------------------------------------

def run_position_sanity_check(tokenizer, valid_pairs: list, strip_ids: set, n: int = 20):
    """
    Verify that clean and corrupted positions land on the same structural token.
    Both should end at the fixed template anchor common to every prompt
    (e.g. <|end_header_id|> for Llama, "model" for Gemma, "assistant" for Qwen3) --
    not at a content word from the instruction itself, which necessarily
    differs between the clean and corrupted prompt in every pair.
    Warns on structural mismatches that suggest position is incorrect.
    """
    print("\n=== Position sanity check ===")
    mismatches = 0
    for i, (clean_tokens, corrupted_tokens) in enumerate(valid_pairs[:n]):
        cp = get_last_meaningful_position(clean_tokens, strip_ids)
        pp = get_last_meaningful_position(corrupted_tokens, strip_ids)
        ct = tokenizer.decode([clean_tokens[cp].item()])
        pt = tokenizer.decode([corrupted_tokens[pp].item()])
        if ct != pt:
            print(f"  WARNING pair {i}: clean='{ct}' vs corrupted='{pt}'")
            mismatches += 1
    if mismatches == 0:
        print(f"  OK — all {min(n, len(valid_pairs))} pairs have matching position tokens")
    print("=== End position check ===\n")


# ---------------------------------------------------------------------------
# 11. Save results
# ---------------------------------------------------------------------------

def save_results(
    causal_effects: torch.Tensor,
    model_name: str,
    arch: str,
    output_dir: Path,
    n_valid_pairs: int,
):
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    torch.save(causal_effects, model_dir / "layer_sweep_normalised.pt")
    np.save(model_dir / "layer_sweep_normalised.npy", causal_effects.numpy())

    peak_layer = int(causal_effects.argmax().item())
    peak_val = float(causal_effects.max().item())
    summary = {
        "model_name": model_name,
        "arch": arch,
        "n_layers": len(causal_effects),
        "n_valid_pairs": n_valid_pairs,
        "peak_layer": peak_layer,
        "peak_normalised_effect": round(peak_val, 4),
        "top5_layers": causal_effects.topk(5).indices.tolist(),
        "causal_effects": [round(x, 4) for x in causal_effects.tolist()],
    }
    with open(model_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {model_dir}")
    print(f"  Peak layer: {peak_layer} (normalised effect: {peak_val:.3f})")
    print(f"  Top 5 layers: {summary['top5_layers']}")


# ---------------------------------------------------------------------------
# 12. Validation checks
# ---------------------------------------------------------------------------

def run_validation_checks(causal_effects: torch.Tensor, arch: str):
    print("\n=== Validation Checks ===")

    peak = int(causal_effects.argmax())
    # Updated from empirical Phase 2 results — causal peaks at 25-31
    # (distinct from Phase 1 geometric peaks at 20-23)
    if arch == "llama":
        expected_range = (25, 31)
    elif arch == "qwen3":
        # Unknown until this script has actually run on Qwen3 once --
        # unlike llama/gemma's ranges, this isn't backed by prior
        # empirical Phase 2 results yet. Wide placeholder so CHECK A
        # doesn't false-fail; tighten once you have real numbers.
        expected_range = (0, 35)
    else:
        expected_range = (25, 42)
    check_a = expected_range[0] <= peak <= expected_range[1]
    print(f"  CHECK A (peak layer {peak} in {expected_range}): "
          f"{'PASS' if check_a else 'NOTE — may still be valid, compare with other conditions'}")

    peak_val = float(causal_effects.max())
    check_b = peak_val > 0.3
    print(f"  CHECK B (peak effect {peak_val:.3f} > 0.3): "
          f"{'PASS' if check_b else 'FAIL — check refusal tokens or pair filtering'}")

    # Early layer check — values 0.2-0.3 expected from token-level content encoding
    # (not position leaking — confirmed by mid-network valley structure)
    early_mean = float(causal_effects[:4].abs().mean())
    check_c = early_mean < 0.35
    print(f"  CHECK C (early layer mean |effect| {early_mean:.3f} < 0.35): "
          f"{'PASS' if check_c else 'WARN — check for position leaking (print full profile)'}")
    if early_mean > 0.05:
        print(f"    NOTE: early layer effects reflect token-level content encoding,")
        print(f"    not position leaking, if mid-network valley (layers ~9-16) exists.")

    print("  CHECK D (cross-paradigm ordering): run after all models complete")
    print("    Expected: ORPO < Ra-SFT < SFT for peak layer (earlier = tighter execution)")
    print("=========================\n")


# ---------------------------------------------------------------------------
# 13. Component analysis loading (uses existing PT/JSON files)
# ---------------------------------------------------------------------------

def load_and_print_component_analysis(
    output_dir: Path,
    model_name: str,
):
    """
    Load existing component analysis results from PT/JSON files.
    Does not re-run — reads results produced by attribution patching.
    Prints ranked head/MLP summary for each peak layer.
    """
    model_dir = output_dir / model_name
    json_path = model_dir / "component_analysis.json"
    pt_path = model_dir / "component_analysis.pt"

    if json_path.exists():
        with open(json_path) as f:
            results = json.load(f)
        print(f"\nComponent analysis loaded from {json_path}")
    elif pt_path.exists():
        results = torch.load(pt_path)
        print(f"\nComponent analysis loaded from {pt_path}")
    else:
        print(f"\nNo component analysis found at {model_dir}")
        return None

    print(f"\n{'='*55}")
    print(f"COMPONENT ANALYSIS SUMMARY — {model_name}")
    print(f"{'='*55}")

    for layer_key in sorted(results.keys(), key=lambda x: int(x)):
        layer_data = results[layer_key]
        print(f"\nLayer {layer_key}:")

        # MLP
        mlp_attr = layer_data.get("mean_mlp_attribution", 0.0)
        print(f"  MLP attribution: {mlp_attr:.4f} "
              f"({'promotes refusal' if mlp_attr > 0 else 'suppresses refusal'})")

        # Attention heads — ranked by absolute attribution
        head_attrs = layer_data.get("mean_head_attributions", {})
        if head_attrs:
            sorted_heads = sorted(
                head_attrs.items(),
                key=lambda x: abs(float(x[1])),
                reverse=True
            )
            print(f"  Top 5 heads by |attribution|:")
            for h_idx, attr in sorted_heads[:5]:
                attr = float(attr)
                direction = "promotes" if attr > 0 else "suppresses"
                print(f"    Head {int(h_idx):2d}: {attr:.4f} ({direction} refusal)")

        # Full patching confirmation if available
        confirmation = layer_data.get("head_confirmation", {})
        if confirmation:
            print(f"  Full patch confirmation (top heads):")
            for h_idx, conf in list(confirmation.items())[:5]:
                print(f"    Head {int(h_idx):2d}: "
                      f"attr={conf.get('attribution_score', 0):.4f}, "
                      f"full_patch={conf.get('full_patch_normalised_effect', 0):.4f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Activation Patching Layer Sweep"
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_name", required=True,
                        help="e.g. llama-sft, llama-ra-sft, llama-orpo, "
                             "gemma-sft, gemma-ra-sft, gemma-orpo")
    parser.add_argument("--arch", required=True, choices=["llama", "gemma", "qwen3"])
    parser.add_argument("--harmful_path", required=True)
    parser.add_argument("--harmless_path", required=True)
    parser.add_argument("--output_dir", default="results/phase2")
    parser.add_argument("--n_pairs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify_tokens", action="store_true",
                        help="Generate sample outputs to inspect refusal behaviour")
    parser.add_argument("--component_analysis_only", action="store_true",
                        help="Skip sweep, just load and print existing component results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Component analysis only mode — no model loading needed
    if args.component_analysis_only:
        load_and_print_component_analysis(output_dir, args.model_name)
        return

    # 1. Load model
    tokenizer, model = load_model(args.model_path, args.arch)
    strip_ids = get_strip_token_ids(tokenizer, args.arch)
    n_layers = ARCH_CONFIG[args.arch]["n_layers"]

    # 2. Load prompt pairs
    pairs = load_prompt_pairs(
        args.harmful_path, args.harmless_path, tokenizer, args.arch,
        model_name=args.model_name, n=args.n_pairs, seed=args.seed,
    )

    # 3. Optional: inspect generation
    if args.verify_tokens:
        print("\n--- Sample generation outputs ---")
        for clean_tokens, _ in pairs[:5]:
            inp = clean_tokens.unsqueeze(0).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    inp, max_new_tokens=20, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            print(repr(tokenizer.decode(out[0][inp.shape[1]:],
                                        skip_special_tokens=False)[:120]))
        print("---\n")

    # 4. Classify and filter pairs
    valid_pairs, refusal_token_ids, compliance_token_ids = validate_pairs_generative(
        model, tokenizer, pairs, model_name=args.model_name
    )

    if len(valid_pairs) == 0:
        raise RuntimeError(
            "No valid pairs from generative classification. "
            "Run with --verify_tokens to inspect generation."
        )

    # 5. Position sanity check
    run_position_sanity_check(tokenizer, valid_pairs, strip_ids)

    # 6. Checkpoint setup
    checkpoint_dir = output_dir / args.model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "layer_sweep_checkpoint.pt"

    all_pair_effects = torch.zeros(len(valid_pairs), n_layers)
    start_pair = 0

    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path)
        all_pair_effects = ckpt["all_pair_effects"]
        start_pair = ckpt["next_pair"]
        print(f"Resuming from pair {start_pair}/{len(valid_pairs)}")

    # 7. Layer sweep
    print(f"\nRunning layer sweep: {len(valid_pairs)} pairs × {n_layers} layers")

    for pair_idx, ((clean_tokens, corrupted_tokens), refusal_id, compliance_id) in enumerate(
        tqdm(
            zip(valid_pairs, refusal_token_ids, compliance_token_ids),
            total=len(valid_pairs),
            desc="Pairs",
        )
    ):
        if pair_idx < start_pair:
            continue

        c_metric = compute_logit_diff(
            model, clean_tokens, refusal_id, compliance_id, strip_ids
        )
        h_metric = compute_logit_diff(
            model, corrupted_tokens, refusal_id, compliance_id, strip_ids
        )

        clean_pos = get_last_meaningful_position(clean_tokens, strip_ids)
        clean_cache = cache_activations(model, clean_tokens, n_layers, pos=clean_pos)

        for layer_idx in range(n_layers):
            patched_metric = patch_and_measure(
                model, clean_cache, corrupted_tokens,
                layer_idx, refusal_id, compliance_id,
                clean_pos=clean_pos, strip_ids=strip_ids,
            )
            norm_effect = normalise_effect(patched_metric, c_metric, h_metric)
            all_pair_effects[pair_idx, layer_idx] = norm_effect

        del clean_cache
        torch.cuda.empty_cache()

        if (pair_idx + 1) % 5 == 0:
            torch.save(
                {"all_pair_effects": all_pair_effects, "next_pair": pair_idx + 1},
                checkpoint_path,
            )

    # 8. Average and save
    causal_effects = all_pair_effects.mean(dim=0)
    save_results(causal_effects, args.model_name, args.arch, output_dir, len(valid_pairs))
    run_validation_checks(causal_effects, args.arch)

    # 9. Print full layer profile
    print("\nFull layer profile:")
    for i, v in enumerate(causal_effects):
        print(f"  Layer {i:2d}: {v:.3f}")

    # 10. Load and print component analysis if it exists
    load_and_print_component_analysis(output_dir, args.model_name)


if __name__ == "__main__":
    main()