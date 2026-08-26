"""
Phase 2: Visualisation — causal effect heatmaps and cross-paradigm comparison plots.

Usage:
    python phase2_visualisation.py \
        --results_dir results/phase2 \
        --arch llama \
        --output_dir plots/phase2

    # Or for the four-panel cross-architecture figure:
    python phase2_visualisation.py \
        --results_dir results/phase2 \
        --arch both \
        --phase1_dir results/phase1 \
        --output_dir plots/phase2
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

# Colours consistent with Phase 1
PARADIGM_COLORS = {
    "sft": "#2196F3",      # blue
    "ra_sft": "#FF9800",   # orange
    "orpo": "#4CAF50",     # green
}
PARADIGM_LABELS = {
    "sft": "SFT",
    "ra_sft": "Ra-SFT",
    "orpo": "ORPO",
}

# Phase 1 geometric peak layers (annotate for cross-method comparison)
PHASE1_PEAKS = {
    "llama": {"sft": 21, "ra_sft": 29, "orpo": 28},
    "gemma": {"sft": None, "orpo": None},  # fill in from Phase 1 results
}


def load_causal_effects(results_dir: Path, arch: str) -> dict:
    """Load normalised causal effect tensors for all available paradigms."""
    results = {}
    paradigms = ["sft", "ra_sft", "orpo"] if arch == "llama" else ["sft", "orpo"]

    for paradigm in paradigms:
        npy_path = results_dir / paradigm / "layer_sweep_normalised.npy"
        pt_path = results_dir / paradigm / "layer_sweep_normalised.pt"

        if npy_path.exists():
            results[paradigm] = np.load(npy_path)
        elif pt_path.exists():
            results[paradigm] = torch.load(pt_path).numpy()
        else:
            print(f"  Warning: no results found for {arch}/{paradigm}")

    return results


def load_phase1_magnitudes(phase1_dir: Path, arch: str) -> dict:
    """Load Phase 1 normalised magnitude profiles if available."""
    results = {}
    if phase1_dir is None:
        return results

    paradigms = ["sft", "ra_sft", "orpo"] if arch == "llama" else ["sft", "orpo"]
    for paradigm in paradigms:
        for fname in ["normalised_magnitudes.npy", "refusal_direction_magnitude.npy"]:
            p = phase1_dir / paradigm / fname
            if p.exists():
                results[paradigm] = np.load(p)
                break

    return results


def plot_causal_heatmap_single_arch(
    causal_effects: dict,
    arch: str,
    phase1_magnitudes: dict = None,
    save_path: Path = None,
    title_suffix: str = "",
):
    """
    Line plot of normalised causal effect per layer, one line per paradigm.
    Optionally overlays Phase 1 magnitude profiles (dashed).
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    n_layers = None
    for paradigm, effects in causal_effects.items():
        n_layers = len(effects)
        x = np.arange(n_layers)
        ax.plot(
            x, effects,
            color=PARADIGM_COLORS.get(paradigm, "gray"),
            label=PARADIGM_LABELS.get(paradigm, paradigm),
            linewidth=2.0,
            marker="o",
            markersize=3,
        )

        # Annotate Phase 1 geometric peak
        arch_peaks = PHASE1_PEAKS.get(arch, {})
        peak_layer = arch_peaks.get(paradigm)
        if peak_layer is not None:
            ax.axvline(
                peak_layer,
                color=PARADIGM_COLORS.get(paradigm, "gray"),
                linestyle="--",
                alpha=0.4,
                linewidth=1.0,
            )

    if phase1_magnitudes:
        for paradigm, mags in phase1_magnitudes.items():
            # Normalise Phase 1 magnitudes to [0, 1] for overlay
            mags_norm = (mags - mags.min()) / (mags.max() - mags.min() + 1e-8)
            ax.plot(
                np.arange(len(mags_norm)),
                mags_norm,
                color=PARADIGM_COLORS.get(paradigm, "gray"),
                linestyle=":",
                alpha=0.5,
                linewidth=1.5,
                label=f"{PARADIGM_LABELS.get(paradigm, paradigm)} (Phase 1, norm.)",
            )

    arch_label = "Llama-3.1-8B" if arch == "llama" else "Gemma-2-9B"
    ax.set_title(f"Activation Patching — Causal Effect by Layer\n{arch_label}{title_suffix}", fontsize=13)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Normalised Causal Effect", fontsize=11)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="-")
    ax.set_xlim(0, n_layers - 1)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, ax


def plot_four_panel(
    llama_phase1: dict,
    llama_phase2: dict,
    gemma_phase1: dict,
    gemma_phase2: dict,
    save_path: Path = None,
):
    """
    Four-panel figure (paper centrepiece):
      Top left:     Llama Phase 1 normalised magnitude
      Top right:    Llama Phase 2 causal effect
      Bottom left:  Gemma Phase 1 normalised magnitude
      Bottom right: Gemma Phase 2 causal effect

    Annotated with peak layers per paradigm.
    """
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

    panel_data = [
        (llama_phase1, "Llama-3.1-8B — Phase 1 (Geometric)", True),
        (llama_phase2, "Llama-3.1-8B — Phase 2 (Causal)", False),
        (gemma_phase1, "Gemma-2-9B — Phase 1 (Geometric)", True),
        (gemma_phase2, "Gemma-2-9B — Phase 2 (Causal)", False),
    ]
    arch_list = ["llama", "llama", "gemma", "gemma"]

    for ax, (data, panel_title, is_phase1), arch in zip(axes, panel_data, arch_list):
        for paradigm, vals in data.items():
            if vals is None:
                continue
            vals = np.array(vals)
            if is_phase1:
                # Normalise for display
                vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
            n_layers = len(vals)
            x = np.arange(n_layers)
            ax.plot(
                x, vals,
                color=PARADIGM_COLORS.get(paradigm, "gray"),
                label=PARADIGM_LABELS.get(paradigm, paradigm),
                linewidth=1.8,
                marker="o",
                markersize=2.5,
            )

            # Annotate peak layer
            peak_layer = int(np.argmax(vals))
            peak_val = vals[peak_layer]
            ax.annotate(
                f"L{peak_layer}",
                xy=(peak_layer, peak_val),
                xytext=(peak_layer + 0.5, peak_val + 0.03),
                fontsize=7,
                color=PARADIGM_COLORS.get(paradigm, "gray"),
                alpha=0.85,
            )

        ax.set_title(panel_title, fontsize=10)
        ax.set_xlabel("Layer", fontsize=9)
        y_label = "Normalised Magnitude (a.u.)" if is_phase1 else "Normalised Causal Effect"
        ax.set_ylabel(y_label, fontsize=9)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Refusal Circuit Structure: Geometric (Phase 1) vs Causal (Phase 2)\n"
        "Normalised profiles across training paradigms",
        fontsize=12,
        y=1.01,
    )
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_component_heatmap(component_results: dict, model_name: str, save_path: Path = None):
    """
    Bar chart of attn vs MLP attribution scores at peak layers.
    component_results: {layer_idx: {mean_attn_attribution, mean_mlp_attribution}}
    """
    layers = sorted(int(k) for k in component_results.keys())
    attn_vals = [component_results[str(l)]["mean_attn_attribution"] for l in layers]
    mlp_vals = [component_results[str(l)]["mean_mlp_attribution"] for l in layers]

    x = np.arange(len(layers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width/2, attn_vals, width, label="Attention", color="#2196F3", alpha=0.8)
    ax.bar(x + width/2, mlp_vals, width, label="MLP", color="#FF9800", alpha=0.8)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Attribution Score")
    ax.set_title(f"Component Attribution at Peak Layers — {model_name}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 2 Visualisation")
    parser.add_argument("--results_dir", required=True, help="Base results/phase2 directory")
    parser.add_argument("--arch", choices=["llama", "gemma", "both"], default="llama")
    parser.add_argument("--phase1_dir", default=None, help="Base results/phase1 directory for overlay")
    parser.add_argument("--output_dir", default="plots/phase2")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    phase1_dir = Path(args.phase1_dir) if args.phase1_dir else None

    if args.arch in ("llama", "both"):
        llama_p2 = load_causal_effects(results_dir / "llama", "llama")
        llama_p1 = load_phase1_magnitudes(phase1_dir / "llama" if phase1_dir else None, "llama") if phase1_dir else {}

        fig, _ = plot_causal_heatmap_single_arch(
            llama_p2, "llama",
            phase1_magnitudes=llama_p1 if llama_p1 else None,
            save_path=output_dir / "llama_causal_profile.png",
        )
        plt.close(fig)

    if args.arch in ("gemma", "both"):
        gemma_p2 = load_causal_effects(results_dir / "gemma", "gemma")
        gemma_p1 = load_phase1_magnitudes(phase1_dir / "gemma" if phase1_dir else None, "gemma") if phase1_dir else {}

        fig, _ = plot_causal_heatmap_single_arch(
            gemma_p2, "gemma",
            phase1_magnitudes=gemma_p1 if gemma_p1 else None,
            save_path=output_dir / "gemma_causal_profile.png",
        )
        plt.close(fig)

    if args.arch == "both":
        # Four-panel paper figure
        llama_p2_data = load_causal_effects(results_dir / "llama", "llama")
        gemma_p2_data = load_causal_effects(results_dir / "gemma", "gemma")
        llama_p1_data = load_phase1_magnitudes(phase1_dir / "llama" if phase1_dir else None, "llama") if phase1_dir else {}
        gemma_p1_data = load_phase1_magnitudes(phase1_dir / "gemma" if phase1_dir else None, "gemma") if phase1_dir else {}

        fig = plot_four_panel(
            llama_p1_data, llama_p2_data,
            gemma_p1_data, gemma_p2_data,
            save_path=output_dir / "four_panel_phase1_phase2.png",
        )
        plt.close(fig)

    # Component heatmaps for any model with component_analysis.json
    for model_dir in results_dir.glob("*/"):
        comp_path = model_dir / "component_analysis.json"
        if comp_path.exists():
            with open(comp_path) as f:
                comp_data = json.load(f)
            fig = plot_component_heatmap(
                comp_data,
                model_dir.name,
                save_path=output_dir / f"component_attribution_{model_dir.name}.png",
            )
            plt.close(fig)


if __name__ == "__main__":
    main()
