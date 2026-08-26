"""
Bootstrap resampling for ASR / ORR confidence intervals.

Unifies bootstrap CI computation for every eval pipeline in this project:

  - WildJailbreak: reads the per-item {"prompt", "response", "unsafe"} labels
    that wildjailbreak.py produces (both a normal run and a
    --skip_generation re-judge run).
  - StrongREJECT: reads the per-item {"model", "jailbreak", "is_unsafe"}
    (or "asr_score") rows that run_strongreject.py produces.
  - XSTest: reads the per-item "prompt_type"/"over_refusal" columns that
    run_xstest.py's per-model CSV export produces.

For each, it:

  1. Bootstraps a CI on the rate (ASR for WildJailbreak/StrongREJECT, ORR for
     XSTest) for each checkpoint/condition individually (percentile method,
     resampling items with replacement).
  2. Computes the Wilson score interval as a cheap closed-form cross-check
     on the same proportion.
  3. (WildJailbreak only) Runs a PAIRED bootstrap for any checkpoint
     comparisons you specify (e.g. llama_sft vs llama_orpo) -- paired
     because both checkpoints were evaluated on the identical prompts, so
     resampling the same indices for both sides (rather than treating them
     as two independent samples) is the correct design and gives tighter,
     more honest CIs on the difference. StrongREJECT/XSTest inputs aren't
     guaranteed to share an identical, aligned prompt set across models, so
     paired comparisons are not offered for them here.

Does NOT pool across eval sets (StrongREJECT vs WildJailbreak vs XSTest) or
across runs that used different generation/judging pipelines -- each input
type is bootstrapped and reported independently, and results are written to
separate top-level keys in the output JSON.

Example -- WildJailbreak + StrongREJECT only (no XSTest), Llama checkpoints:

    python bootstrap_asr.py \
        --results_json ../results/WildJailbreak/rejudge_results.json \
        --strongreject ../results/StrongREJECT/llama/llama-base_scores.json \
                       ../results/StrongREJECT/llama/safety-sft_scores.json \
                       ../results/StrongREJECT/llama/safety-orpo_scores.json \
                       ../results/StrongREJECT/llama/safety-ra-sft_scores.json \
        --output_path bootstrap_llama_wj_sr.json

Omitting --xstest simply skips that section (run_xstest / run_strongreject /
run_wildjailbreak each only fire when their input flag is passed).
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


# ── Wilson score interval (closed-form cross-check) ─────────────────────────
def wilson_ci(successes, n, alpha=0.05):
    if n == 0:
        return (float("nan"), float("nan"))
    z = _z_for_alpha(alpha)
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return (max(0.0, center - margin), min(1.0, center + margin))


def _z_for_alpha(alpha):
    # Two-sided critical value. Covers the common cases without requiring
    # scipy; extend the table if you need alphas beyond these.
    table = {0.10: 1.645, 0.05: 1.960, 0.01: 2.576}
    if alpha in table:
        return table[alpha]
    raise ValueError(f"No z-value tabulated for alpha={alpha}; "
                      f"add one to _z_for_alpha or use alpha in {list(table)}")


# ── Single-group bootstrap ───────────────────────────────────────────────────
def bootstrap_ci(labels, n_boot=10000, alpha=0.05, seed=42):
    """
    Percentile bootstrap CI for a binary-outcome rate (ASR or ORR), plus a
    Wilson score interval cross-check.
    labels: list/array of bool (True = unsafe/attack-successful, or
            over-refused, depending on the metric).
    """
    arr = np.asarray(labels, dtype=float)
    n = len(arr)
    if n == 0:
        return None
    point_estimate = arr.mean()

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_rates = arr[idx].mean(axis=1)

    lo_pct, hi_pct = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    ci_low, ci_high = np.percentile(boot_rates, [lo_pct, hi_pct])

    wci_low, wci_high = wilson_ci(int(arr.sum()), n, alpha)

    return {
        "n": n,
        "n_positive": int(arr.sum()),
        "point_estimate": float(point_estimate),
        "bootstrap_ci": (float(ci_low), float(ci_high)),
        "bootstrap_se": float(boot_rates.std(ddof=1)) if n_boot > 1 else float("nan"),
        "wilson_ci": (float(wci_low), float(wci_high)),
        "n_boot": n_boot,
        "alpha": alpha,
    }


# ── Paired bootstrap for checkpoint comparisons (WildJailbreak) ─────────────
def paired_bootstrap_diff(labels_a, labels_b, n_boot=10000, alpha=0.05,
                          seed=42):
    """
    Paired bootstrap on rate_a - rate_b, resampling the SAME indices for both
    checkpoints each replicate (valid because both were run on the
    identical prompt set -- this is not two independent binomials).
    """
    a = np.asarray(labels_a, dtype=float)
    b = np.asarray(labels_b, dtype=float)
    if len(a) != len(b):
        raise ValueError(
            f"Paired comparison requires equal-length, aligned label "
            f"arrays (same prompt set) -- got {len(a)} vs {len(b)}. "
            f"If these came from different max_samples runs they are "
            f"not directly comparable."
        )
    n = len(a)
    point_diff = a.mean() - b.mean()

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)

    lo_pct, hi_pct = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    ci_low, ci_high = np.percentile(boot_diffs, [lo_pct, hi_pct])
    excludes_zero = (ci_low > 0) or (ci_high < 0)

    return {
        "n": n,
        "point_diff": float(point_diff),
        "bootstrap_ci": (float(ci_low), float(ci_high)),
        "excludes_zero": bool(excludes_zero),
        "n_boot": n_boot,
        "alpha": alpha,
    }


# ── Loading: WildJailbreak ──────────────────────────────────────────────────
def load_wildjailbreak(paths):
    """
    Merge one or more results JSONs (each shaped {model_name: {"examples":
    [{"prompt","response","unsafe"}, ...], ...}}) into {model_name: [bool]}.
    Works whether that's one combined file (a normal wildjailbreak.py run)
    or several per-model files (wildjailbreak.py --skip_generation run
    with --models one at a time).
    """
    labels_by_model = {}
    for path in paths:
        data = json.loads(Path(path).read_text())
        for name, result in data.items():
            if "examples" not in result:
                continue
            flags = [bool(e["unsafe"]) for e in result["examples"]]
            if name in labels_by_model:
                print(f"  Warning: '{name}' appears in multiple input "
                      f"files -- using the version from {path}")
            labels_by_model[name] = flags
    return labels_by_model


# ── Loading: StrongREJECT ────────────────────────────────────────────────────
def load_strongreject(path):
    """Load StrongREJECT-style JSON. Groups by (model, jailbreak) -> list of is_unsafe."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    groups = defaultdict(list)
    for row in data:
        model = row.get("model", "UNKNOWN_MODEL")
        jailbreak = row.get("jailbreak", "UNKNOWN_ATTACK")
        if "is_unsafe" in row:
            outcome = bool(row["is_unsafe"])
        elif "asr_score" in row:
            outcome = bool(row["asr_score"])
        else:
            raise KeyError(
                f"Row {row.get('idx')} has neither 'is_unsafe' nor 'asr_score' -- "
                "check the field name in your actual file."
            )
        groups[(model, jailbreak)].append(outcome)
    return groups


# ── Loading: XSTest ──────────────────────────────────────────────────────────
def load_xstest(path, model_name=None):
    """Load XSTest-style CSV. Groups by (model, prompt_type) -> list of over_refusal."""
    groups = defaultdict(list)
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = model_name or row.get("model", "UNKNOWN_MODEL")
            prompt_type = row.get("prompt_type", "ALL")
            raw = row.get("over_refusal", "")
            outcome = str(raw).strip().lower() in ("true", "1", "yes")
            groups[(model, prompt_type)].append(outcome)
    return groups


# ── Reporting helpers for grouped (model, condition) metrics ───────────────
def report_grouped(groups, metric_name, n_boot, alpha, seed):
    """Bootstrap each (model, condition) group and print + return results."""
    print(f"\n{'='*72}")
    print(f"{metric_name} bootstrap results (n_boot={n_boot})")
    print(f"{'-'*72}")
    print(f"{'Model':<22}{'Condition':<18}{'N':>6}{'Point':>8}{'CI_low':>8}{'CI_high':>8}")
    results = {}
    for (model, condition), values in sorted(groups.items()):
        res = bootstrap_ci(values, n_boot=n_boot, alpha=alpha, seed=seed)
        if res is None:
            continue
        results[f"{model}::{condition}"] = res
        print(
            f"{model:<22}{condition:<18}{res['n']:>6}"
            f"{res['point_estimate']*100:>7.1f}%"
            f"{res['bootstrap_ci'][0]*100:>7.1f}%{res['bootstrap_ci'][1]*100:>7.1f}%"
        )
    print(f"{'='*72}")
    return results


def report_aggregate(groups, metric_name, n_boot, alpha, seed):
    """Aggregate across all conditions per model (e.g. overall ASR per model)."""
    by_model = defaultdict(list)
    for (model, _condition), values in groups.items():
        by_model[model].extend(values)

    print(f"\n{metric_name} aggregated across all conditions, per model")
    print(f"{'-'*72}")
    print(f"{'Model':<22}{'N':>6}{'Point':>8}{'CI_low':>8}{'CI_high':>8}")
    results = {}
    for model, values in sorted(by_model.items()):
        res = bootstrap_ci(values, n_boot=n_boot, alpha=alpha, seed=seed)
        if res is None:
            continue
        results[model] = res
        print(
            f"{model:<22}{res['n']:>6}"
            f"{res['point_estimate']*100:>7.1f}%"
            f"{res['bootstrap_ci'][0]*100:>7.1f}%{res['bootstrap_ci'][1]*100:>7.1f}%"
        )
    return results


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    # WildJailbreak inputs
    parser.add_argument("--results_json", type=str, nargs="*", default=[],
                        help="One or more WildJailbreak results JSON files "
                             "(combined or per-model) to load labels from")
    parser.add_argument("--models", type=str, nargs="*", default=None,
                        help="Which WildJailbreak checkpoints to compute "
                             "single-model CIs for. Default: all found.")
    parser.add_argument("--pairs", type=str, nargs="*", default=None,
                        help="Paired WildJailbreak comparisons as "
                             "'modelA:modelB', e.g. "
                             "llama_sft:llama_orpo llama_ra_sft:llama_orpo")

    # StrongREJECT inputs
    parser.add_argument("--strongreject", nargs="*", default=[],
                        help="StrongREJECT JSON file(s)")

    # XSTest inputs
    parser.add_argument("--xstest", nargs="*", default=[],
                        help="XSTest CSV file(s)")
    parser.add_argument("--xstest-model-names", nargs="*", default=[],
                         help="If XSTest CSVs are per-model and lack a "
                              "'model' column, give one name per --xstest "
                              "file, same order.")

    # Shared bootstrap params
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", type=str,
                        default="bootstrap_results.json")
    return parser.parse_args()


def run_wildjailbreak(args, output):
    labels_by_model = load_wildjailbreak(args.results_json)
    if not labels_by_model:
        raise ValueError(
            "No models with 'examples' found in the given --results_json file(s)."
        )

    models = args.models or sorted(labels_by_model.keys())
    missing = [m for m in models if m not in labels_by_model]
    if missing:
        raise KeyError(f"Requested model(s) not found in input: {missing}. "
                        f"Available: {sorted(labels_by_model.keys())}")

    print(f"\n{'='*72}")
    print(f"{'Model':<18} {'N':>6} {'ASR':>8} {'Bootstrap 95% CI':>20} "
          f"{'Wilson 95% CI':>20}")
    print(f"{'-'*72}")

    single_results = {}
    for name in models:
        res = bootstrap_ci(labels_by_model[name], args.n_boot,
                           args.alpha, args.seed)
        single_results[name] = res
        boot_str = f"[{res['bootstrap_ci'][0]*100:.2f}, {res['bootstrap_ci'][1]*100:.2f}]"
        wilson_str = f"[{res['wilson_ci'][0]*100:.2f}, {res['wilson_ci'][1]*100:.2f}]"
        print(f"{name:<18} {res['n']:>6} {res['point_estimate']*100:>7.2f}% "
              f"{boot_str:>20} {wilson_str:>20}")
    print(f"{'='*72}")

    pair_results = {}
    if args.pairs:
        print(f"\nPaired comparisons (resampling shared prompt indices):")
        print(f"{'-'*72}")
        for pair_str in args.pairs:
            a_name, b_name = pair_str.split(":")
            for nm in (a_name, b_name):
                if nm not in labels_by_model:
                    raise KeyError(f"'{nm}' from pair '{pair_str}' not "
                                   f"found in input.")
            res = paired_bootstrap_diff(
                labels_by_model[a_name], labels_by_model[b_name],
                args.n_boot, args.alpha, args.seed
            )
            pair_results[pair_str] = res
            sig_flag = "  <- CI excludes 0" if res["excludes_zero"] else ""
            print(f"  {a_name} - {b_name}: "
                  f"diff={res['point_diff']*100:+.2f}pp  "
                  f"95% CI=[{res['bootstrap_ci'][0]*100:+.2f}, "
                  f"{res['bootstrap_ci'][1]*100:+.2f}]{sig_flag}")
        print(f"{'-'*72}")

    output["wildjailbreak"] = {"single_model": single_results, "pairwise": pair_results}


def run_strongreject(args, output):
    all_sr_groups = defaultdict(list)
    for path in args.strongreject:
        g = load_strongreject(path)
        for k, v in g.items():
            all_sr_groups[k].extend(v)
    per_condition = report_grouped(all_sr_groups, "ASR (StrongREJECT, is_unsafe)",
                                   args.n_boot, args.alpha, args.seed)
    aggregate = report_aggregate(all_sr_groups, "ASR (StrongREJECT, all attacks combined)",
                                 args.n_boot, args.alpha, args.seed)
    output["strongreject"] = {"per_condition": per_condition, "aggregate_per_model": aggregate}


def run_xstest(args, output):
    all_xs_groups = defaultdict(list)
    names = args.xstest_model_names or [None] * len(args.xstest)
    if len(names) != len(args.xstest):
        raise ValueError("--xstest-model-names must match --xstest in count, or be omitted")
    for path, name in zip(args.xstest, names):
        g = load_xstest(path, model_name=name)
        for k, v in g.items():
            all_xs_groups[k].extend(v)
    per_condition = report_grouped(all_xs_groups, "ORR (XSTest, over_refusal)",
                                   args.n_boot, args.alpha, args.seed)
    aggregate = report_aggregate(all_xs_groups, "ORR (XSTest, all prompt types combined)",
                                 args.n_boot, args.alpha, args.seed)
    output["xstest"] = {"per_condition": per_condition, "aggregate_per_model": aggregate}


def main():
    args = parse_args()

    if not (args.results_json or args.strongreject or args.xstest):
        raise ValueError(
            "Nothing to do -- pass at least one of --results_json "
            "(WildJailbreak), --strongreject, or --xstest."
        )

    output = {}
    if args.results_json:
        run_wildjailbreak(args, output)
    if args.strongreject:
        run_strongreject(args, output)
    if args.xstest:
        run_xstest(args, output)

    Path(args.output_path).write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {args.output_path}")


if __name__ == "__main__":
    main()
