"""
Bootstrap resampling of component patching results, by prompt pair, to assess
causal component-ranking stability -- answers UDyf's request to "report
robustness of causal component rankings across prompt subsets and random seeds."

REQUIRES the .pt checkpoint file saved by component_patching.py (NOT just the
.json summary), because it needs per-pair effect values to resample from.
The .json only has aggregated mean/std -- there is no way to bootstrap from
that alone. Check that component_patching.pt exists alongside your .json
before running this; if it doesn't, the per-pair effects were not retained
and this would require re-running component_patching.py itself (real GPU time).

WHAT THIS DOES
--------------
For each of --n-boot iterations:
  1. Resample prompt pairs with replacement (same n as original, e.g. n=201)
  2. Recompute each component's mean_normalised_effect on that resample
  3. Rank all components by |mean_normalised_effect| (descending)
Then reports:
  - 95% CI on each component's mean effect (percentile bootstrap)
  - How often each component lands in the top-K ranking across resamples
    (rank stability -- the actual answer to "robustness of rankings")
  - Spearman rank correlation between the original ranking and each
    bootstrap resample's ranking, summarised as mean +/- std across resamples

Usage:
    python bootstrap_component_patching.py \
        --pt_path results/phase2/gemma-orpo/component_patching.pt \
        --json_path results/phase2/gemma-orpo/component_patching.json \
        --n_boot 1000 --top_k 5
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from scipy.stats import spearmanr


def load_component_tensors(pt_path):
    """
    Returns: dict[(layer, component_name)] -> 1D tensor of per-pair effects.
    component_name is "mlp" or "head_{idx}".
    """
    raw = torch.load(pt_path, map_location="cpu", weights_only=False)
    out = {}
    for layer_str, layer_tensors in raw.items():
        if "mlp_norm_effects" in layer_tensors:
            out[(layer_str, "mlp")] = layer_tensors["mlp_norm_effects"]
        if "head_tensors" in layer_tensors:
            for key, tensor in layer_tensors["head_tensors"].items():
                # key like "head_7_norm_effects"
                head_idx = key.replace("head_", "").replace("_norm_effects", "")
                out[(layer_str, f"head_{head_idx}")] = tensor
    return out


def ranking_from_effects(effects_dict):
    """effects_dict: {component_key: scalar mean effect}. Returns list of
    component_keys sorted by |effect| descending."""
    return sorted(effects_dict.keys(), key=lambda k: abs(effects_dict[k]), reverse=True)


def bootstrap_component_stability(component_tensors, n_boot=1000, top_k=5, seed=42):
    rng = random.Random(seed)

    # sanity check: all components should have the same n_pairs (same pairs
    # were used across components within a run). Warn if not.
    ns = {k: len(v) for k, v in component_tensors.items()}
    n_unique = set(ns.values())
    if len(n_unique) > 1:
        print(f"WARNING: components have differing n_pairs: {ns}")
        print("Resampling will use each component's own n -- rankings may not "
              "be strictly comparable across components with different n.")

    # Original point-estimate ranking (mean over all pairs, as originally computed)
    original_effects = {k: float(v.mean()) for k, v in component_tensors.items()}
    original_ranking = ranking_from_effects(original_effects)

    # For rank-correlation we need a fixed universe of ranks (1..len(components))
    all_keys = list(component_tensors.keys())
    orig_rank_map = {k: r for r, k in enumerate(original_ranking)}
    orig_rank_vector = [orig_rank_map[k] for k in all_keys]

    top_k_counts = defaultdict(int)
    boot_effect_samples = defaultdict(list)
    spearman_rhos = []

    for _ in range(n_boot):
        boot_effects = {}
        for key, tensor in component_tensors.items():
            n = tensor.shape[0]
            idx = torch.tensor([rng.randrange(n) for _ in range(n)])
            resample = tensor[idx]
            boot_effects[key] = float(resample.mean())
            boot_effect_samples[key].append(boot_effects[key])

        boot_ranking = ranking_from_effects(boot_effects)
        for comp in boot_ranking[:top_k]:
            top_k_counts[comp] += 1

        boot_rank_map = {k: r for r, k in enumerate(boot_ranking)}
        boot_rank_vector = [boot_rank_map[k] for k in all_keys]
        rho, _ = spearmanr(orig_rank_vector, boot_rank_vector)
        spearman_rhos.append(rho)

    # Assemble CI per component
    ci_results = {}
    for key, samples in boot_effect_samples.items():
        samples_sorted = sorted(samples)
        lo = samples_sorted[int(0.025 * n_boot)]
        hi = samples_sorted[min(int(0.975 * n_boot), n_boot - 1)]
        ci_results[key] = {
            "point": original_effects[key],
            "ci_low": lo,
            "ci_high": hi,
            "top_k_frequency": top_k_counts[key] / n_boot,
        }

    return {
        "original_ranking": original_ranking,
        "ci_results": ci_results,
        "spearman_mean": float(np.mean(spearman_rhos)),
        "spearman_std": float(np.std(spearman_rhos)),
        "n_boot": n_boot,
        "top_k": top_k,
    }


def print_report(result, model_name=""):
    print(f"\n{'=' * 78}")
    print(f"Component patching bootstrap stability -- {model_name}")
    print(f"{'=' * 78}")
    print(f"Original top-{result['top_k']} ranking (by |mean normalised effect|):")
    for i, comp in enumerate(result["original_ranking"][: result["top_k"]]):
        layer, name = comp
        print(f"  {i+1}. Layer {layer}, {name}")

    print(f"\nSpearman rank correlation (original vs. bootstrap resample), "
          f"across {result['n_boot']} resamples:")
    print(f"  mean = {result['spearman_mean']:.3f}, std = {result['spearman_std']:.3f}")
    print("  (1.0 = identical ranking every resample; lower = rankings shift under resampling)")

    print(f"\n{'Layer':<8}{'Component':<12}{'Point':>10}{'CI_low':>10}{'CI_high':>10}"
          f"{'Top-'+str(result['top_k'])+' freq':>14}")
    sorted_comps = sorted(
        result["ci_results"].items(), key=lambda kv: abs(kv[1]["point"]), reverse=True
    )
    for (layer, comp), stats in sorted_comps:
        print(
            f"{layer:<8}{comp:<12}{stats['point']:>+10.4f}{stats['ci_low']:>+10.4f}"
            f"{stats['ci_high']:>+10.4f}{stats['top_k_frequency']*100:>13.1f}%"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pt_path", required=True, help="Path to component_patching.pt")
    parser.add_argument("--json_path", required=False,
                         help="Optional: component_patching.json, for cross-checking "
                              "that point estimates match the original saved run")
    parser.add_argument("--model_name", default="")
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pt_path = Path(args.pt_path)
    if not pt_path.exists():
        raise FileNotFoundError(
            f"{pt_path} not found. This script requires the .pt checkpoint with "
            "per-pair effect tensors -- the .json summary alone (means/stds only) "
            "is not sufficient for resampling. If you only have the .json, the "
            "per-pair effects were not retained on disk and component_patching.py "
            "would need to be re-run to regenerate them."
        )

    component_tensors = load_component_tensors(pt_path)
    print(f"Loaded {len(component_tensors)} components from {pt_path}")

    if args.json_path:
        with open(args.json_path, "r", encoding="utf-8") as f:
            original_json = json.load(f)
        # quick sanity cross-check on one component
        first_layer = next(iter(original_json))
        json_mlp_mean = original_json[first_layer]["mlp"]["mean_normalised_effect"]
        pt_mlp_mean = float(component_tensors[(first_layer, "mlp")].mean())
        diff = abs(json_mlp_mean - pt_mlp_mean)
        if diff > 1e-4:
            print(
                f"WARNING: layer {first_layer} MLP mean differs between .json "
                f"({json_mlp_mean:.6f}) and recomputed from .pt ({pt_mlp_mean:.6f}). "
                "Check the .pt file matches the .json run before trusting results."
            )
        else:
            print(f"Sanity check passed: .pt tensors match .json summary for layer {first_layer} MLP.")

    result = bootstrap_component_stability(
        component_tensors, n_boot=args.n_boot, top_k=args.top_k, seed=args.seed
    )
    print_report(result, model_name=args.model_name)


if __name__ == "__main__":
    main()