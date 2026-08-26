# phase1_refusal_direction.py
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from huggingface_hub import HfApi, login
import torch.nn.functional as F
import subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────

HF_TOKEN        = ""
HF_RESULTS_REPO = "HoangCuongNguyen/refusal-direction-results"  # can reuse phase0c repo
ARDITI_DATASET  = "./refusal_direction/dataset"
OUTPUT_DIR      = Path("./results/phase1")
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS          = list(range(32))  # all layers for Llama-3.1-8B
N_PAIRS         = 256
N_STABILITY_SUBSETS = 5  # for direction stability check
RANDOM_SEED      = 42
FORCE_RERUN      = False   # set True to regenerate from scratch

# Add to config section
RUN_EARLY_LAYER_DIAGNOSTIC = True
EARLY_DIAG_LAYERS          = [3, 4, 5]
# All three variants — run this script once per MODEL entry
# or pass as CLI arg (see __main__ at bottom)
MODEL_CONFIGS = {
    # Fine-tuned variants
    # "llama-sft": {
    #     "path":    "models/llama-sft",
    #     "arch":    "llama",
    #     "is_base": False,
    # },
    # "llama-ra-sft": {
    #     "path":    "models/llama-ra-sft",
    #     "arch":    "llama",
    #     "is_base": False,
    # },
    # "llama-orpo": {
    #     "path":    "models/llama-orpo",
    #     "arch":    "llama",
    #     "is_base": False,
    # },
    # # Base models — for Du et al. contextualisation
    # "llama-base": {
    #     "path":    "models/llama-base",
    #     "arch":    "llama",
    #     "is_base": True,
    # },
    # "gemma-base": {
    #     "path":    "models/gemma-base",
    #     "arch":    "gemma",
    #     "is_base": True,
    # },
    # # Gemma fine-tuned
    # "gemma-sft": {
    #     "path":    "models/gemma-sft",
    #     "arch":    "gemma",
    #     "is_base": False,
    # },
    # "gemma-ra-sft": {
    #     "path":    "models/gemma-ra-sft",
    #     "arch":    "gemma",
    #     "is_base": False,
    # },
    # "gemma-orpo": {
    #     "path":    "models/gemma-orpo",
    #     "arch":    "gemma",
    #     "is_base": False,
    # },
    "qwen3-base": {
        "path":    "Qwen/Qwen3-8B",
        "arch":    "qwen3",
        "is_base": True,
    },
    "qwen3-sft": {
        "path":    "models/qwen-sft",
        "arch":    "qwen3",
        "is_base": False,
    },
    "qwen3-ra-sft": {
        "path":    "models/qwen-ra-sft",
        "arch":    "qwen3",
        "is_base": False,
    },
    "qwen3-orpo": {
        "path":    "models/qwen-orpo",
        "arch":    "qwen3",
        "is_base": False,
    },
}

# Add to config / architecture definitions
ARCHITECTURE_CONFIGS = {
    "llama": {
        "n_layers":   32,
        "d_model":    4096,
        "peak_layers": [27, 28, 29, 30, 31],
        "peak_layer":  28,   # middle of peak cluster
    },
    "gemma": {
        "n_layers":   42,
        "d_model":    3584,
        "peak_layers": [37, 38, 39, 40, 41],
        "peak_layer":  40,   # middle of peak cluster
    },
    "qwen3": {
        "n_layers":   36,     # verified: Qwen/Qwen3-8B config.json
        "d_model":    4096,   # verified: Qwen/Qwen3-8B config.json
        "peak_layers": None,  # unknown until Phase 1 runs once for Qwen3 —
        "peak_layer":  None,  # this is an empirical finding, not an architectural constant.
                               # After the first run, read summary.json for each qwen3-*
                               # variant and fill these in before rerunning the
                               # cross-variant comparison section (plots null-guard
                               # this already, so it's safe to leave None meanwhile).
    },
}

# Plot display labels
VARIANT_LABELS = {
    # "llama-base":       "Llama base model",
    # "llama-sft":        "Llama Safety SFT",
    # "llama-ra-sft":     "Llama Safety Ra-SFT",
    # "llama-orpo":       "Llama Safety ORPO",
    "qwen-base":       "Qwen3 base model",
    "qwen-sft":        "Qwen3 SFT",
    "qwen-ra-sft":     "Qwen3 Ra-SFT",
    "qwen-orpo":       "Qwen3 ORPO",
}

VARIANT_COLORS = {
    "qwen-base": "#9C27B0",
    "qwen-sft":  "#2196F3",
    "qwen-ra-sft":  "#4CAF50",
    "qwen-orpo":    "#FF5722",
    # "qwen3-base": "#607D8B",
    # "qwen3-sft":  "#00BCD4",
    # "qwen3-ra-sft":  "#8BC34A",
    # "qwen3-orpo":    "#FFC107",
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "logs").mkdir(exist_ok=True)

# ── Shutdown ──────────────────────────────────────────────────────────────────

def shutdown(reason: str):
    print(f"\n[SHUTDOWN] Reason: {reason}")
    try:
        push_to_hf(final=True)
    except Exception as e:
        print(f"[SHUTDOWN] HF push failed: {e}")
    print("[SHUTDOWN] Powering off in 60 seconds.")
    print("[SHUTDOWN] Cancel with: sudo shutdown -c")
    subprocess.run(["sudo", "shutdown", "-h", "+1"])
    sys.exit(0)

# ── HF persistence ────────────────────────────────────────────────────────────

def push_to_hf(final: bool = False):
    api = HfApi(token=HF_TOKEN)
    try:
        api.create_repo(
            HF_RESULTS_REPO,
            repo_type="dataset",
            private=True,
            exist_ok=True,
        )
    except Exception:
        pass
    label = "final" if final else "incremental"
    print(f"[HF] Pushing {label} results...")
    api.upload_folder(
        folder_path=str(OUTPUT_DIR),
        repo_id=HF_RESULTS_REPO,
        repo_type="dataset",
        path_in_repo="phase1",
    )
    print("[HF] Done.")

# ── Dataset ───────────────────────────────────────────────────────────────────

def load_arditi_pairs(arditi_path: str, n_pairs: int):
    """
    Load harmful/harmless pairs from Arditi's splits directory.
    Uses train split for direction extraction — keeps test split
    clean for downstream evaluation.

    Each JSON file is a list of dicts with at minimum a "instruction"
    key containing the prompt string. Checking load_dataset.py
    would confirm exact schema but this is the standard format
    across all processed files.
    """
    splits_dir  = os.path.join(arditi_path, "splits")
    h_path      = os.path.join(splits_dir, "harmful_train.json")
    hn_path     = os.path.join(splits_dir, "harmless_train.json")

    if not os.path.exists(h_path):
        raise FileNotFoundError(
            f"Expected {h_path} — confirm splits/ directory exists"
        )
    if not os.path.exists(hn_path):
        raise FileNotFoundError(
            f"Expected {hn_path} — confirm splits/ directory exists"
        )

    with open(h_path)  as f: harmful_raw  = json.load(f)
    with open(hn_path) as f: harmless_raw = json.load(f)

    print(f"[DATA] Train split sizes: "
          f"{len(harmful_raw)} harmful, {len(harmless_raw)} harmless")

    # Extract prompt strings — handle both possible schemas:
    # {"instruction": "..."} or plain string list
    def extract_prompts(data):
        if not data:
            raise ValueError("Empty dataset")
        if isinstance(data[0], dict):
            # Try common key names
            for key in ["instruction", "prompt", "text", "goal"]:
                if key in data[0]:
                    return [item[key] for item in data]
            # If none match, print keys to diagnose
            print(f"[DATA] Available keys: {list(data[0].keys())}")
            raise KeyError(
                f"No recognised prompt key in dataset. "
                f"Keys found: {list(data[0].keys())}"
            )
        elif isinstance(data[0], str):
            return data
        else:
            raise TypeError(f"Unexpected data format: {type(data[0])}")

    harmful_prompts  = extract_prompts(harmful_raw)[:n_pairs]
    harmless_prompts = extract_prompts(harmless_raw)[:n_pairs]

    print(f"[DATA] Using {len(harmful_prompts)} pairs for "
          f"direction extraction")
    print(f"[DATA] Example harmful:  {harmful_prompts[0][:80]}...")
    print(f"[DATA] Example harmless: {harmless_prompts[0][:80]}...")

    return harmful_prompts, harmless_prompts

# ── Model loading ─────────────────────────────────────────────────────────────

# def load_model(model_path: str, arch: str) -> HookedTransformer:
#     """
#     Architecture-aware TransformerLens loading.

#     Llama: TL handles fine-tuned HF paths directly.
#     Gemma: Load via HF first then pass to TL to avoid
#            name resolution issues with fine-tuned checkpoints.
#     """
#     if arch == "llama":
#         print(f"  [LOAD] Loading Llama via TransformerLens...")
#         model = HookedTransformer.from_pretrained(
#             model_path,
#             dtype=torch.bfloat16,
#             device=DEVICE,
#         )

#     elif arch == "gemma":
#         print(f"  [LOAD] Loading Gemma via HuggingFace first...")
#         hf_model = AutoModelForCausalLM.from_pretrained(
#             model_path,
#             torch_dtype=torch.bfloat16,
#             device_map=DEVICE,
#         )
#         tokenizer = AutoTokenizer.from_pretrained(model_path)

#         print(f"  [LOAD] Converting to HookedTransformer...")
#         # Pass the reference architecture name so TL knows
#         # the config, then override weights with fine-tuned ones
#         model = HookedTransformer.from_pretrained(
#             "google/gemma-2-9b-it",
#             hf_model=hf_model,
#             tokenizer=tokenizer,
#             dtype=torch.bfloat16,
#             device=DEVICE,
#         )
#         del hf_model
#         torch.cuda.empty_cache()

#     else:
#         raise ValueError(f"Unknown architecture: {arch}")

#     model.eval()
#     print(f"  [LOAD] d_model={model.cfg.d_model}, "
#           f"n_layers={model.cfg.n_layers}")

#     # Sanity check against expected config
#     expected = ARCHITECTURE_CONFIGS[arch]
#     assert model.cfg.n_layers == expected["n_layers"], (
#         f"Expected {expected['n_layers']} layers for {arch}, "
#         f"got {model.cfg.n_layers}"
#     )
#     assert model.cfg.d_model == expected["d_model"], (
#         f"Expected d_model={expected['d_model']} for {arch}, "
#         f"got {model.cfg.d_model}"
#     )

#     return model

# def load_model(model_path: str, arch: str) -> HookedTransformer:
    
#     # Architecture reference names for TransformerLens
#     TL_ARCH_NAMES = {
#         "llama": "meta-llama/Llama-3.1-8B",
#         "gemma": "google/gemma-2-9b",
#     }
    
#     print(f"  [LOAD] Loading {arch} from local path: {model_path}")
    
#     hf_model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         torch_dtype=torch.bfloat16,
#         device_map=DEVICE,
#     )
#     tokenizer = AutoTokenizer.from_pretrained(model_path)

#     print(f"  [LOAD] Converting to HookedTransformer "
#           f"(reference: {TL_ARCH_NAMES[arch]})...")
#     model = HookedTransformer.from_pretrained(
#         TL_ARCH_NAMES[arch],
#         hf_model=hf_model,
#         tokenizer=tokenizer,
#         dtype=torch.bfloat16,
#         device=DEVICE,
#     )

#     del hf_model
#     torch.cuda.empty_cache()

#     model.eval()
#     print(f"  [LOAD] d_model={model.cfg.d_model}, "
#           f"n_layers={model.cfg.n_layers}")

#     # Sanity check
#     expected = ARCHITECTURE_CONFIGS[arch]
#     assert model.cfg.n_layers == expected["n_layers"], (
#         f"Expected {expected['n_layers']} layers, "
#         f"got {model.cfg.n_layers}"
#     )
#     assert model.cfg.d_model == expected["d_model"], (
#         f"Expected d_model={expected['d_model']}, "
#         f"got {model.cfg.d_model}"
#     )

#     return model

def load_model(model_path: str, arch: str):
    """
    Returns (model, tokenizer) using native HF.
    No TransformerLens conversion — avoids peak VRAM spike.
    """
    expected = ARCHITECTURE_CONFIGS[arch]

    print(f"  [LOAD] {model_path}")
    print(f"  [LOAD] Expected: {expected['n_layers']} layers, "
          f"d_model={expected['d_model']}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    # Sanity check via model config
    n_layers = model.config.num_hidden_layers
    d_model  = model.config.hidden_size
    print(f"  [LOAD] Actual: {n_layers} layers, d_model={d_model}")

    assert n_layers == expected["n_layers"], (
        f"Layer count mismatch: expected {expected['n_layers']}, "
        f"got {n_layers}"
    )
    assert d_model == expected["d_model"], (
        f"d_model mismatch: expected {expected['d_model']}, "
        f"got {d_model}"
    )

    vram_used = torch.cuda.memory_allocated() / 1e9
    print(f"  [LOAD] VRAM after load: {vram_used:.1f}GB")

    return model, tokenizer

def load_model_native(model_path: str):
    """
    Skip TransformerLens entirely for Phase 1.
    Native HF hooks are sufficient for residual stream extraction.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    return model, tokenizer




def format_prompt_native(prompt: str, arch: str,
                          is_base: bool = False) -> str:
    if is_base:
        return prompt    # no template for base models
    
    if arch == "llama":
        return (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    elif arch == "gemma":
        return (
            "<start_of_turn>user\n"
            f"{prompt}"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
    elif arch == "qwen3":
        # Empty <think></think> stub matches enable_thinking=False,
        # the convention used across the SFT/ORPO/Ra-SFT training scripts.
        # If your eval pipeline (StrongREJECT/WildJailbreak/XSTest) instead
        # prompts Qwen3 with enable_thinking=True at inference, switch this
        # to match — direction extraction should reflect the same input
        # distribution the model sees at eval time, not just at training time.
        return (
            "<|im_start|>user\n"
            f"{prompt}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )
    raise ValueError(f"Unknown arch: {arch}")


def extract_all_layers_native(
    model,
    tokenizer,
    prompts:  list,
    layers:   list,
    arch:     str,
    is_base:  bool = False,    # new, defaults False for safety
):
    acts = {l: [] for l in layers}
    hooks = []

    def make_hook(layer_idx):
        
       def hook_fn(module, input, output):
           # Robustly handle both Tuple and Tensor outputs
           if isinstance(output, tuple):
               hidden = output[0]
           else:
               hidden = output
            
        # Safety check: ensures we have [Batch, Seq, Dim]
           if hidden.ndim == 3:
               acts[layer_idx].append(
                   hidden[0, -1, :].detach().cpu().float()
               )
           else:
               # If it's already 2D [Seq, Dim], just take the last token
               acts[layer_idx].append(
                   hidden[-1, :].detach().cpu().float()
               )
       return hook_fn

    for l in layers:
        h = model.model.layers[l].register_forward_hook(
            make_hook(l)
        )
        hooks.append(h)

    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            if i % 50 == 0:
                print(f"  [{i}/{len(prompts)}]")

            inputs = tokenizer(
                format_prompt_native(prompt, arch,
                                     is_base=is_base),  # passed through
                return_tensors="pt",
            ).to("cuda")

            model(**inputs)

            del inputs
            torch.cuda.empty_cache()

    for h in hooks:
        h.remove()

    return {l: torch.stack(acts[l]) for l in layers}

def run_single_model(
    model_name:       str,
    model_path:       str,
    arch:             str,
    is_base:          bool,
    harmful_prompts:  list,
    harmless_prompts: list,
):
    model_dir  = OUTPUT_DIR / model_name
    model_dir.mkdir(exist_ok=True)
    checkpoint = model_dir / "directions.pt"
    
    if checkpoint.exists() and not FORCE_RERUN:
        print(f"[{model_name}] Checkpoint found — loading.")
        saved = torch.load(checkpoint)
        return saved["directions"], saved["magnitude_profile"]

    arch_cfg = ARCHITECTURE_CONFIGS[arch]
    layers   = list(range(arch_cfg["n_layers"]))

    print(f"\n[{model_name}] arch={arch}, is_base={is_base}")

    try:
        model, tokenizer = load_model(model_path, arch)
    except Exception as e:
        raise RuntimeError(
            f"Model load failed for {model_name}: {e}"
        )

    print(f"[{model_name}] Extracting harmful activations...")
    harmful_acts = extract_all_layers_native(
        model, tokenizer,
        harmful_prompts, layers, arch,
        is_base=is_base    # passed through
    )
    print(f"[{model_name}] Extracting harmless activations...")
    harmless_acts = extract_all_layers_native(
        model, tokenizer,
        harmless_prompts, layers, arch,
        is_base=is_base    # passed through
    )
    

    # Free model — activations on CPU, GPU free for next model
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    vram_after = torch.cuda.memory_allocated() / 1e9
    print(f"[{model_name}] VRAM after model free: {vram_after:.1f}GB")

    # Everything below unchanged from previous version
    print(f"[{model_name}] Computing magnitude profile...")
    mag_profile = direction_magnitude_profile(
        harmful_acts, harmless_acts, layers
    )

    print(f"[{model_name}] Computing directions...")
    directions = compute_directions_all_layers(
        harmful_acts, harmless_acts, layers
    )

    print(f"[{model_name}] Computing stability...")
    stability = direction_stability(
        harmful_acts, harmless_acts, layers,
        n_subsets=N_STABILITY_SUBSETS,
        seed=RANDOM_SEED,
    )


    # torch.save({
    #     "directions":        directions,
    #     "magnitude_profile": mag_profile,
    #     "stability":         stability,
    #     "arch":              arch,
    #     "layers":            layers,
    # }, checkpoint)
    print(f"[{model_name}] Computing representation cosine similarity...")
    rep_cosine = harmful_harmless_cosine_similarity(
        harmful_acts, harmless_acts, layers
    )

    mag_profile_raw        = direction_magnitude_profile(
        harmful_acts, harmless_acts, layers
    )

    mag_profile_normalised = direction_magnitude_profile_normalised(
        rep_cosine, layers
    )

     # Save into checkpoint 
    torch.save({
        "directions":             directions,
        "magnitude_profile":      mag_profile_raw,
        "magnitude_profile_norm": mag_profile_normalised,  # new
        "stability":              stability,
        "rep_cosine":             rep_cosine,
        "arch":                   arch,
        "layers":                 layers,
    }, checkpoint)
    peak_layer  =  int(max(mag_profile_normalised,
                           key=mag_profile_normalised.get)),
    top5_layers = sorted(
        mag_profile_normalised,
        key=mag_profile_normalised.get,
        reverse=True
    )[:5],

   
    # torch.save({
    #    "directions":        directions,
    #    "magnitude_profile": mag_profile,
    #    "stability":         stability,
    #    "rep_cosine":        rep_cosine,   # new
    #    "arch":              arch,
    #    "layers":            layers,
    # }, checkpoint)
    # peak_layer  = int(max(mag_profile, key=mag_profile.get))
    # top5_layers = sorted(
    #     mag_profile, key=mag_profile.get, reverse=True
    # )[:5]

    summary = {
        "model_name":        model_name,
        "model_path":        model_path,
        "arch":              arch,
        "n_layers":          len(layers),
        "peak_layer":        peak_layer,
        "top5_layers":       top5_layers,
        "magnitude_profile": {
            str(l): v for l, v in mag_profile.items()
        },
        "magnitude_profile_norm": {str(l): v
                                   for l, v in mag_profile_normalised.items()},
        "stability_summary": {
            str(l): {
                "mean": stability[l]["mean"],
                "min":  stability[l]["min"],
            }
            for l in layers[::4]
        },
        "rep_cosine_top5": {
            str(l): {
                "cos_sim":              rep_cosine[l]["cos_sim_harmful_harmless"],
                 "normalised_magnitude": rep_cosine[l]["normalised_magnitude"],
            }
            for l in layers[::4]
        },
    }
    with open(model_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{model_name}] Peak layer: {peak_layer}")
    print(f"[{model_name}] Top 5 layers: {top5_layers}")

    del harmful_acts, harmless_acts
    gc.collect()
    torch.cuda.empty_cache()

    return directions, mag_profile

# ── Tokenisation ──────────────────────────────────────────────────────────────

# def format_prompt(model: HookedTransformer,
#                   prompt: str,
#                   arch: str) -> torch.Tensor:
#     """
#     Apply correct chat template per architecture.
#     prepend_bos=False because both templates include
#     the BOS token explicitly.
#     """
#     if arch == "llama":
#         formatted = (
#             "<|begin_of_text|>"
#             "<|start_header_id|>user<|end_header_id|>\n\n"
#             f"{prompt}"
#             "<|eot_id|>"
#             "<|start_header_id|>assistant<|end_header_id|>\n\n"
#         )
#     elif arch == "gemma":
#         formatted = (
#             "<bos>"
#             "<start_of_turn>user\n"
#             f"{prompt}"
#             "<end_of_turn>\n"
#             "<start_of_turn>model\n"
#         )
#     else:
#         raise ValueError(f"Unknown architecture: {arch}")

#     return model.to_tokens(formatted, prepend_bos=False)

# ── Activation extraction ─────────────────────────────────────────────────────




# def extract_all_layers(
#     model:   HookedTransformer,
#     prompts: list,
#     layers:  list,
#     arch:    str,
# ) -> dict:
#     """
#     One forward pass per prompt, cache last-token residual stream
#     at ALL layers simultaneously.

#     Last token position is where the model begins generating the
#     assistant response — the refusal decision forms here.

#     Returns {layer_idx: tensor[n_prompts, d_model]}
#     """
#     names_filter = [f"blocks.{l}.hook_resid_post" for l in layers]
#     acts = {l: [] for l in layers}

#     with torch.no_grad():
#         for i, prompt in enumerate(prompts):
#             if i % 50 == 0:
#                 print(f"  [{i}/{len(prompts)}]")

#             tokens = format_prompt(model, prompt, arch)

#             _, cache = model.run_with_cache(
#                 tokens,
#                 names_filter=names_filter,
#                 return_type=None,
#             )

#             for l in layers:
#                 act = cache[
#                     f"blocks.{l}.hook_resid_post"
#                 ][0, -1, :].cpu().float()
#                 acts[l].append(act)

#             del cache
#             torch.cuda.empty_cache()

#     return {l: torch.stack(acts[l]) for l in layers}

# ── Direction computation ─────────────────────────────────────────────────────

def difference_in_means(
    harmful_acts:  torch.Tensor,
    harmless_acts: torch.Tensor,
) -> torch.Tensor:
    """Unit-normalised difference-in-means refusal direction."""
    direction = harmful_acts.mean(0) - harmless_acts.mean(0)
    return direction / direction.norm()


def compute_directions_all_layers(
    harmful_acts:  dict,
    harmless_acts: dict,
    layers:        list,
) -> dict:
    """Returns {layer: unit direction [d_model]}"""
    return {
        l: difference_in_means(harmful_acts[l], harmless_acts[l])
        for l in layers
    }


def direction_magnitude_profile(
    harmful_acts:  dict,
    harmless_acts: dict,
    layers:        list,
) -> dict:
    """
    Unnormalised difference magnitude per layer.
    Primary Phase 1 output — reveals which layers carry
    the strongest refusal signal.
    """
    profile = {}
    for l in layers:
        diff = harmful_acts[l].mean(0) - harmless_acts[l].mean(0)
        profile[l] = diff.norm().item()
    return profile

def harmful_harmless_cosine_similarity(
    harmful_acts:  dict,
    harmless_acts: dict,
    layers:        list,
) -> dict:
    """
    Cosine similarity between mean harmful and mean harmless
    activations per layer.
    
    High value = representations are close together in residual
    stream space = DIM vector has low norm not because refusal
    is weak but because representations are compressed.
    
    Distinguishes Ra-SFT Reading A (genuine lower refusal signal)
    from Reading B (compressed representation space artefact).
    """
    results = {}
    for l in layers:
        mean_harmful  = harmful_acts[l].mean(0)
        mean_harmless = harmless_acts[l].mean(0)
        cos_sim = F.cosine_similarity(
            mean_harmful.unsqueeze(0),
            mean_harmless.unsqueeze(0),
        ).item()
        
        # Also compute relative magnitude normalised by
        # mean activation norm — removes scale differences
        norm_harmful  = mean_harmful.norm().item()
        norm_harmless = mean_harmless.norm().item()
        mean_norm     = (norm_harmful + norm_harmless) / 2
        
        diff = mean_harmful - mean_harmless
        normalised_magnitude = diff.norm().item() / mean_norm
        
        results[l] = {
            "cos_sim_harmful_harmless": cos_sim,
            "normalised_magnitude":     normalised_magnitude,
            "mean_norm_harmful":        norm_harmful,
            "mean_norm_harmless":       norm_harmless,
        }
    
    return results

def direction_magnitude_profile_normalised(
    rep_cosine: dict,
    layers:     list,
) -> dict:
    return {
        l: rep_cosine[l]["normalised_magnitude"]
        for l in layers
        if l in rep_cosine
    }


def direction_stability(
    harmful_acts:  dict,
    harmless_acts: dict,
    layers:        list,
    n_subsets:     int,
    subset_size:   int = 64,
    seed:          int = 42,
) -> dict:


    """
    Bootstrap stability: compute direction on n_subsets random subsets,
    measure pairwise cosine similarity across subsets per layer.

    High mean cosine sim (>0.95) confirms direction is stable and
    not driven by a small number of idiosyncratic prompts.
    Worth one sentence in the paper as a robustness check.
    """
    n = harmful_acts[layers[0]].shape[0]
    # Guard against requesting more than available
    subset_size = min(subset_size, n // 2)
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    subsets = [
        torch.randperm(n, generator=generator)[:subset_size]
        for _ in range(n_subsets)
    ]

    stability = {}
    for l in layers:
        subset_dirs = [
            difference_in_means(
                harmful_acts[l][idx],
                harmless_acts[l][idx],
            )
            for idx in subsets
        ]

        sims = [
            F.cosine_similarity(
                subset_dirs[i].unsqueeze(0),
                subset_dirs[j].unsqueeze(0),
            ).item()
            for i in range(len(subset_dirs))
            for j in range(i + 1, len(subset_dirs))
        ]

        stability[l] = {
            "mean": float(np.mean(sims)),
            "std":  float(np.std(sims)),
            "min":  float(np.min(sims)),
        }

    return stability

def find_crossover_layer(
    base_profile:     dict,
    finetuned_profile: dict,
    layers:           list,
) -> int:
    """
    Find layer where fine-tuned normalised magnitude
    first exceeds base model magnitude.
    This marks the onset of safety-specific computation.
    """
    for l in layers:
        if (l in base_profile and
            l in finetuned_profile and
            finetuned_profile[l] > base_profile[l]):
            return l
    return None


# ── Cross-variant comparison ──────────────────────────────────────────────────

def pairwise_cosine_matrix(
    directions_per_variant: dict,
    layers: list,
) -> dict:
    """
    Compute per-layer pairwise cosine similarity matrix across variants.
    Only call this within a single architecture — comparing Llama
    and Gemma directions is not meaningful.

    Returns {layer: {"matrix": [[...]], "variants": [...]}}
    """
    variants = list(directions_per_variant.keys())
    results  = {}

    for l in layers:
        n   = len(variants)
        mat = np.zeros((n, n))
        for i, v1 in enumerate(variants):
            for j, v2 in enumerate(variants):
                mat[i, j] = F.cosine_similarity(
                    directions_per_variant[v1][l].unsqueeze(0),
                    directions_per_variant[v2][l].unsqueeze(0),
                ).item()
        results[l] = {"matrix": mat.tolist(), "variants": variants}

    return results

# ── Plotting ──────────────────────────────────────────────────────────────────

# def plot_magnitude_profiles(
#     profiles_per_variant: dict,
#     layers:     list,
#     output_dir: Path,
#     suffix:     str = "",
# ):
#     """
#     Line plot: refusal direction magnitude vs layer, one line per variant.
#     Secondary Phase 1 visual — shows where in the network refusal
#     signal is concentrated per training objective.
#     """
#     fig, ax = plt.subplots(figsize=(11, 5))

#     for variant, profile in profiles_per_variant.items():
#         # Only plot layers that exist in this profile
#         plot_layers = [l for l in layers if l in profile]
#         mags = [profile[l] for l in plot_layers]
#         ax.plot(
#             plot_layers, mags,
#             label=VARIANT_LABELS.get(variant, variant),
#             color=VARIANT_COLORS.get(variant, "gray"),
#             linewidth=2,
#             marker="o",
#             markersize=3,
#         )

#     ax.set_xlabel("Layer", fontsize=12)
#     ax.set_ylabel("Refusal Direction Magnitude\n"
#                   "(Unnormalised DIM)", fontsize=11)
#     arch_label = suffix.upper() if suffix else "All Variants"
#     ax.set_title(
#         f"Layer-Resolved Refusal Direction Magnitude — {arch_label}",
#         fontsize=13,
#     )
#     ax.legend(fontsize=10)
#     ax.grid(True, alpha=0.3)
#     ax.set_xticks(layers[::2])

#     plt.tight_layout()
#     fname = f"magnitude_profile{'_' + suffix if suffix else ''}.png"
#     path  = output_dir / fname
#     plt.savefig(path, dpi=150)
#     plt.close()
#     print(f"[PLOT] Saved → {path}")


def plot_magnitude_profiles(
    profiles_per_variant: dict,
    layers:      list,
    output_dir:  Path,
    suffix:      str  = "",
    normalised:  bool = False,
    base_keys:   list = None,    # new — keys to plot as dashed
):
    fig, ax = plt.subplots(figsize=(11, 5))

    for variant, profile in profiles_per_variant.items():
        plot_layers = [l for l in layers if l in profile]
        mags        = [profile[l] for l in plot_layers]

        # Base models as dashed lines — visually distinct
        is_base = (base_keys and variant in base_keys)
        ax.plot(
            plot_layers, mags,
            label=VARIANT_LABELS.get(variant, variant),
            color=VARIANT_COLORS.get(variant, "gray"),
            linewidth=2,
            marker="o" if not is_base else "s",
            markersize=3,
            linestyle="--" if is_base else "-",
            alpha=0.7 if is_base else 1.0,
        )

    ax.set_xlabel("Layer", fontsize=12)

    if normalised:
        ax.set_ylabel(
            "Normalised Refusal Direction Magnitude\n"
            r"$\|\mu_{harm} - \mu_{harmless}\|$"
            r"$ / \bar{\|\mu\|}$",
            fontsize=11,
        )
        title_suffix = "(Normalised)"
    else:
        ax.set_ylabel(
            "Refusal Direction Magnitude\n"
            "(Unnormalised — scale reference only)",
            fontsize=11,
        )
        title_suffix = "(Unnormalised)"

    arch_label = suffix.upper() if suffix else "All Variants"
    ax.set_title(
        f"Layer-Resolved Refusal Direction Magnitude"
        f" — {arch_label}\n{title_suffix}",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(layers[::2])

    plt.tight_layout()
    norm_tag = "_norm" if normalised else "_raw"
    fname    = (f"magnitude_profile{norm_tag}"
                f"{'_' + suffix if suffix else ''}.png")
    path     = output_dir / fname
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved → {path}")


def check_early_layer_divergence(
    models_dict:      dict,
    tokenizers_dict:  dict,
    harmless_prompts: list,
    layers:           list = [3, 4, 5],
    n_prompts:        int  = 50,
) -> dict:
    """
    Diagnostic: are early-layer differences safety-specific
    or general across prompt types?
    
    Run on harmless prompts only — if divergence persists
    on harmless prompts it's not safety-specific (Interp B/C).
    If it disappears, it's training objective fingerprint (Interp A).
    
    Returns {layer: {pair: cosine_sim}}
    """
    
    results = {}
    sample     = harmless_prompts[:n_prompts]
    # Accumulate mean activations per model per layer
    # {model_name: {layer: mean_tensor}}
    all_acts   = {name: {} for name in models_dict}

    # ── One model at a time ───────────────────────────────────────
    for name, model in models_dict.items():
        tokenizer = tokenizers_dict[name]
        print(f"\n[EARLY DIAG] Extracting layer acts for {name}...")

        # Storage keyed by layer — this is what make_hook writes to
        layer_storage = {l: [] for l in layers}

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output

                if hidden.ndim == 3:
                    layer_storage[layer_idx].append(
                        hidden[0, -1, :].detach().cpu().float()
                    )
                else:
                    layer_storage[layer_idx].append(
                        hidden[-1, :].detach().cpu().float()
                    )
            return hook_fn

        # Register hooks for all diagnostic layers at once
        # so we only need one forward pass per prompt
        hooks = []
        for l in layers:
            h = model.model.layers[l].register_forward_hook(
                make_hook(l)
            )
            hooks.append(h)

        with torch.no_grad():
            for i, prompt in enumerate(sample):
                if i % 10 == 0:
                    print(f"  [{i}/{len(sample)}]")
                inputs = tokenizer(
                    format_prompt_native(
                        prompt,
                        MODEL_CONFIGS[name]["arch"],
                        is_base=MODEL_CONFIGS[name]["is_base"],
                    ),
                    return_tensors="pt",
                ).to("cuda")
                model(**inputs)
                del inputs
                torch.cuda.empty_cache()

        # Remove hooks
        for h in hooks:
            h.remove()

        # Store mean activation per layer for this model
        for l in layers:
            if layer_storage[l]:
                all_acts[name][l] = torch.stack(
                    layer_storage[l]
                ).mean(0)
                print(f"  Layer {l}: captured "
                      f"{len(layer_storage[l])} activations, "
                      f"shape {all_acts[name][l].shape}")
            else:
                print(f"  [WARN] No activations for "
                      f"{name} at layer {l}")
                all_acts[name][l] = None

    # ── Pairwise cosine similarities per layer ────────────────────
    names = list(all_acts.keys())

    for l in layers:
        layer_sims = {}

        for i, n1 in enumerate(names):
            for j, n2 in enumerate(names):
                if i < j:
                    a1 = all_acts[n1].get(l)
                    a2 = all_acts[n2].get(l)

                    if a1 is None or a2 is None:
                        layer_sims[f"{n1}_vs_{n2}"] = float("nan")
                        continue

                    sim = F.cosine_similarity(
                        a1.unsqueeze(0),
                        a2.unsqueeze(0),
                    ).item()
                    layer_sims[f"{n1}_vs_{n2}"] = sim

        results[l] = layer_sims
        print(f"\n[EARLY DIAG] Layer {l} "
              f"cosine sim on harmless prompts:")
        for pair, sim in layer_sims.items():
            print(f"  {pair}: {sim:.4f}")

    return results

def plot_cosine_heatmap(
    cosine_results: dict,
    layers:         list,
    output_dir:     Path,
    suffix:         str = "",
    peak_layers:    list = None,   # new — annotate these cells
):
    if not cosine_results:
        return

    variants    = cosine_results[layers[0]]["variants"]
    pairs = [
        (variants[i], variants[j])
        for i in range(len(variants))
        for j in range(i + 1, len(variants))
    ]
    if not pairs:
        return

    pair_labels = [
        f"{VARIANT_LABELS.get(a, a)} vs "
        f"{VARIANT_LABELS.get(b, b)}"
        for a, b in pairs
    ]

    data = np.zeros((len(pairs), len(layers)))
    for col, l in enumerate(layers):
        mat      = np.array(cosine_results[l]["matrix"])
        var_list = cosine_results[l]["variants"]
        for row, (a, b) in enumerate(pairs):
            i = var_list.index(a)
            j = var_list.index(b)
            data[row, col] = mat[i, j]

    fig, ax = plt.subplots(
        figsize=(14, max(3, len(pairs) * 1.2))
    )
    im = ax.imshow(
        data,
        aspect="auto",
        cmap="RdYlGn",
        vmin=0.0,      # fixed: 0 not 0.7
        vmax=1.0,
    )
    plt.colorbar(im, ax=ax, label="Cosine Similarity")

    # Annotate peak layer cells
    if peak_layers:
        peak_col_indices = [
            i for i, l in enumerate(layers)
            if l in peak_layers
        ]
        for row in range(data.shape[0]):
            for col in peak_col_indices:
                val = data[row, col]
                ax.text(
                    col, row, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=7,
                    color="black" if 0.25 < val < 0.75
                          else "white",
                )

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(
        [str(l) for l in layers], fontsize=7
    )
    ax.set_yticks(range(len(pair_labels)))
    ax.set_yticklabels(pair_labels, fontsize=9)
    ax.set_xlabel("Layer", fontsize=12)
    arch_label = suffix.upper() if suffix else ""
    ax.set_title(
        f"Pairwise Refusal Direction Cosine Similarity by Layer"
        f"{' — ' + arch_label if arch_label else ''}\n"
        f"(Raw residual stream)",
        fontsize=11,
    )

    plt.tight_layout()
    fname = (f"cosine_heatmap"
             f"{'_' + suffix if suffix else ''}.png")
    path  = output_dir / fname
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved → {path}")


def plot_stability(
    stability_per_variant: dict,
    layers:     list,
    output_dir: Path,
    suffix:     str = "",
):
    """
    Line plot of mean bootstrap cosine similarity per layer.
    Confirms direction is not prompt-driven.
    Low priority visual — include in appendix if needed.
    """
    fig, ax = plt.subplots(figsize=(11, 4))

    for variant, stability in stability_per_variant.items():
        plot_layers = [l for l in layers if l in stability]
        means = [stability[l]["mean"] for l in plot_layers]
        stds  = [stability[l]["std"]  for l in plot_layers]

        ax.plot(
            plot_layers, means,
            label=VARIANT_LABELS.get(variant, variant),
            color=VARIANT_COLORS.get(variant, "gray"),
            linewidth=2,
        )
        ax.fill_between(
            plot_layers,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            color=VARIANT_COLORS.get(variant, "gray"),
            alpha=0.15,
        )

    ax.axhline(0.95, color="black", linestyle="--",
               linewidth=1, label="0.95 threshold")
    ax.set_ylim(0.5, 1.05)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Bootstrap Cosine Similarity\n(mean ± std)", fontsize=11)
    ax.set_title(
        "Refusal Direction Stability Across Prompt Subsets",
        fontsize=13,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(layers[::2])

    plt.tight_layout()
    fname = f"stability{'_' + suffix if suffix else ''}.png"
    path  = output_dir / fname
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved → {path}")

def plot_cosine_similarity_profiles(
    cosine_results: dict,
    layers:         list,
    output_dir:     Path,
    suffix:         str  = "",
    peak_layer:     int  = None,   # new — vertical line position
):
    if not cosine_results:
        return

    variants = cosine_results[layers[0]]["variants"]
    pairs    = [
        (variants[i], variants[j])
        for i in range(len(variants))
        for j in range(i + 1, len(variants))
    ]

    base_names = [
        v for v in variants
        if MODEL_CONFIGS.get(v, {}).get("is_base", False)
    ]
    post_names = [
        v for v in variants
        if not MODEL_CONFIGS.get(v, {}).get("is_base", False)
    ]

    intra_post_pairs = [
        (a, b) for a, b in pairs
        if a in post_names and b in post_names
    ]
    base_pairs = [
        (a, b) for a, b in pairs
        if a in base_names or b in base_names
    ]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(16, 5),
        sharey=True,
    )

    # ── Colour and style maps ─────────────────────────────────────────
    # Keyed by frozenset so order doesn't matter
    PAIR_STYLES = {
        # Llama intra-POST
        frozenset({"llama-sft", "llama-ra_sft"}):        {"color": "#E91E63", "label": "Llama Safety SFT vs Llama Safety Ra-SFT"},
        frozenset({"llama-sft", "llama-orpo"}):          {"color": "#2196F3", "label": "Llama Safety SFT vs Llama Safety ORPO"},
        frozenset({"llama-ra-sft", "llama-orpo"}):       {"color": "#4CAF50", "label": "Llama Safety Ra-SFT vs Llama Safety ORPO"},
        # Gemma intra-POST — all three pairs
        frozenset({"gemma-sft", "gemma-ra-sft"}):  {"color": "#E91E63", "label": "Gemma Safety SFT vs Gemma Safety Ra-SFT"},
        frozenset({"gemma-sft", "gemma-orpo"}):    {"color": "#2196F3", "label": "Gemma Safety SFT vs Gemma Safety ORPO"},
        frozenset({"gemma-ra-sft", "gemma-orpo"}): {"color": "#4CAF50", "label": "Gemma Safety Ra-SFT vs Gemma Safety ORPO"},
        # Llama base vs POST
        frozenset({"llama-base", "llama-sft"}):    {"color": "#FF9800", "label": "Llama Base vs Llama Safety SFT"},
        frozenset({"llama-base", "llama-ra-sft"}): {"color": "#9C27B0", "label": "Llama Base vs Llama Safety Ra-SFT"},
        frozenset({"llama-base", "llama-orpo"}):   {"color": "#795548", "label": "Llama Base vs Llama Safety ORPO"},
        # Gemma base vs POST
        frozenset({"gemma-base", "gemma-sft"}):    {"color": "#FF9800", "label": "Gemma Base vs Gemma Safety SFT"},
        frozenset({"gemma-base", "gemma-ra-sft"}): {"color": "#9C27B0", "label": "Gemma Base vs Gemma Safety Ra-SFT"},
        frozenset({"gemma-base", "gemma-orpo"}):   {"color": "#795548", "label": "Gemma Base vs Gemma Safety ORPO"},
    }

    def get_sim_series(a, b):
        sims = []
        for l in layers:
            mat = np.array(cosine_results[l]["matrix"])
            vl  = cosine_results[l]["variants"]
            if a in vl and b in vl:
                sims.append(mat[vl.index(a), vl.index(b)])
            else:
                sims.append(np.nan)
        return sims

    # Left panel — intra-POST
    ax = axes[0]
    for a, b in intra_post_pairs:
        key    = frozenset({a, b})
        style  = PAIR_STYLES.get(key, {"color": "gray", "label": f"{a} vs {b}"})
        color  = style["color"]
        label  = style["label"]
        sims   = get_sim_series(a, b)
        ax.plot(
            layers, sims,
            label=label, color=color,
            linewidth=2, marker="o", markersize=3,
        )
    if peak_layer is not None:
        ax.axvline(
            peak_layer,
            color="gray", linestyle=":",
            linewidth=1.5, alpha=0.8,
            label=f"Peak layer ({peak_layer})",
        )
    # Highlight Gemma-specific local attention architecture region
    if suffix == "gemma":
        ax.axvspan(
            14, 17,
            alpha=0.08, color="orange",
            label="Global attention dominant (L14-17)",
        )
    ax.set_ylim(-0.15, 1.05)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Cosine Similarity", fontsize=12)
    ax.set_title(
        "Intra-POST Variant Pairs\n"
        f"({suffix.upper() if suffix else 'All'})",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(layers[::4])

    # Right panel — base vs POST
    ax = axes[1]
    for a, b in base_pairs:
        key   = frozenset({a, b})
        style = PAIR_STYLES.get(key, {"color": "gray", "label": f"{a} vs {b}"})
        color = style["color"]
        label = style["label"]
        sims  = get_sim_series(a, b)
        ax.plot(
            layers, sims,
            label=label, color=color,
            linewidth=2, marker="o", markersize=3,
        )

    ax.axhline(
        0.0, color="black", linestyle="-",
        linewidth=0.5, alpha=0.3,
    )
    if peak_layer is not None:
        ax.axvline(
            peak_layer,
            color="gray", linestyle=":",
            linewidth=1.5, alpha=0.8,
            label=f"Peak layer ({peak_layer})",
        )
    ax.set_ylim(-0.15, 1.05)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_title(
        "Base vs POST Variants\n"
        "(Du et al. contextualisation)",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(layers[::4])

    arch_label = suffix.upper() if suffix else "All Variants"
    plt.suptitle(
        f"Refusal Direction Cosine Similarity Across Layers"
        f" — {arch_label}",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()

    fname = (f"cosine_similarity_profiles"
             f"{'_' + suffix if suffix else ''}.png")
    path  = output_dir / fname
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved → {path}")

# ── Per-model analysis ────────────────────────────────────────────────────────

# def run_single_model(
#     model_name:       str,
#     model_path:       str,
#     arch:             str,
#     harmful_prompts:  list,
#     harmless_prompts: list,
# ):
#     """
#     Full Phase 1 analysis for one model variant.
#     Resumes from checkpoint if partial run exists.
#     Returns (directions_dict, magnitude_profile_dict).
#     """
#     model_dir  = OUTPUT_DIR / model_name
#     model_dir.mkdir(exist_ok=True)
#     checkpoint = model_dir / "directions.pt"

#     if checkpoint.exists():
#         print(f"[{model_name}] Checkpoint found — loading cached results.")
#         saved = torch.load(checkpoint)
#         return saved["directions"], saved["magnitude_profile"]

#     arch_cfg = ARCHITECTURE_CONFIGS[arch]
#     layers   = list(range(arch_cfg["n_layers"]))

#     print(f"\n[{model_name}] arch={arch}, "
#           f"n_layers={arch_cfg['n_layers']}, "
#           f"d_model={arch_cfg['d_model']}")

#     # Load model
#     try:
#         model = load_model(model_path, arch)
#     except Exception as e:
#         raise RuntimeError(f"Model load failed for {model_name}: {e}")

#     # Extract activations — single pass per prompt across all layers
#     print(f"[{model_name}] Extracting harmful activations "
#           f"({len(harmful_prompts)} prompts)...")
#     harmful_acts = extract_all_layers(
#         model, harmful_prompts, layers, arch
#     )

#     print(f"[{model_name}] Extracting harmless activations "
#           f"({len(harmless_prompts)} prompts)...")
#     harmless_acts = extract_all_layers(
#         model, harmless_prompts, layers, arch
#     )

#     # Free model — activations are on CPU, model no longer needed
#     del model
#     torch.cuda.empty_cache()

#     # Magnitude profile — primary Phase 1 output
#     print(f"[{model_name}] Computing magnitude profile...")
#     mag_profile = direction_magnitude_profile(
#         harmful_acts, harmless_acts, layers
#     )

#     # Unit-normalised directions — for cross-variant comparison
#     print(f"[{model_name}] Computing directions...")
#     directions = compute_directions_all_layers(
#         harmful_acts, harmless_acts, layers
#     )

#     # Bootstrap stability
#     print(f"[{model_name}] Computing direction stability "
#           f"({N_STABILITY_SUBSETS} subsets)...")
#     stability = direction_stability(
#         harmful_acts, harmless_acts, layers,
#         n_subsets=N_STABILITY_SUBSETS,
#     )

#     # Save checkpoint
#     torch.save({
#         "directions":        directions,
#         "magnitude_profile": mag_profile,
#         "stability":         stability,
#         "arch":              arch,
#         "layers":            layers,
#     }, checkpoint)
#     print(f"[{model_name}] Checkpoint saved → {checkpoint}")

#     # Human-readable summary
#     peak_layer  = int(max(mag_profile, key=mag_profile.get))
#     top5_layers = sorted(
#         mag_profile, key=mag_profile.get, reverse=True
#     )[:5]

#     summary = {
#         "model_name":        model_name,
#         "model_path":        model_path,
#         "arch":              arch,
#         "n_layers":          len(layers),
#         "peak_layer":        peak_layer,
#         "top5_layers":       top5_layers,
#         "magnitude_profile": {str(l): v for l, v in mag_profile.items()},
#         "stability_summary": {
#             str(l): {
#                 "mean": stability[l]["mean"],
#                 "min":  stability[l]["min"],
#             }
#             for l in layers[::4]  # every 4th layer to keep file small
#         },
#     }
#     with open(model_dir / "summary.json", "w") as f:
#         json.dump(summary, f, indent=2)

#     # Print layer table
#     print(f"\n[{model_name}] Peak layer: {peak_layer}")
#     print(f"[{model_name}] Top 5 layers: {top5_layers}")
#     print(f"\n  Layer | Magnitude | Stability (mean)")
#     print(  "  ------|-----------|----------------")
#     for l in layers[::4]:
#         print(
#             f"  {l:5d} | "
#             f"{mag_profile[l]:9.4f} | "
#             f"{stability[l]['mean']:.4f}"
#         )

#     del harmful_acts, harmless_acts
#     torch.cuda.empty_cache()

#     return directions, mag_profile

# ── Main ──────────────────────────────────────────────────────────────────────

def main(model_names: list = None):
    login(token=HF_TOKEN)

    if model_names is None:
        model_names = list(MODEL_CONFIGS.keys())

    # Validate requested names
    unknown = [n for n in model_names if n not in MODEL_CONFIGS]
    if unknown:
        print(f"[WARN] Unknown model names: {unknown}")
        print(f"[WARN] Available: {list(MODEL_CONFIGS.keys())}")
        model_names = [n for n in model_names if n in MODEL_CONFIGS]

    print(f"[MAIN] Phase 1 — running: {model_names}")

    # Load Arditi dataset once — shared across all variants
    print("\n[DATA] Loading Arditi dataset...")
    try:
        harmful_prompts, harmless_prompts = load_arditi_pairs(
            ARDITI_DATASET, N_PAIRS
        )
    except Exception as e:
        shutdown(f"Dataset load failed: {e}")

    all_directions         = {}
    all_mag_profiles       = {}
    all_stability          = {}

    for name in model_names:
        cfg = MODEL_CONFIGS[name]
        print(f"\n{'='*60}")
        print(f"[MAIN] Starting: {name} ({cfg['arch']})")
        print(f"{'='*60}")

        try:
            directions, mag_profile = run_single_model(
               model_name=name,
               model_path=cfg["path"],
               arch=cfg["arch"],
               is_base=cfg["is_base"],    # new
               harmful_prompts=harmful_prompts,
               harmless_prompts=harmless_prompts,
            )
            all_directions[name]   = directions
            all_mag_profiles[name] = mag_profile

            # Load stability from saved checkpoint for plotting
            ckpt = torch.load(OUTPUT_DIR / name / "directions.pt")
            all_stability[name] = ckpt["stability"]

            push_to_hf(final=False)

        except Exception as e:
            # Log the failure but continue with remaining models
            print(f"[ERROR] {name} failed: {e}")
            import traceback
            with open(OUTPUT_DIR / "logs" / f"{name}_error.txt", "w") as f:
                traceback.print_exc(file=f)
            print(f"[MAIN] Continuing with remaining models...")
            continue

    if not all_directions:
        shutdown("All models failed — check error logs")

    # ── Cross-variant comparison, separately per architecture ─────────────────
    for arch in ["llama", "gemma", "qwen3"]:
        arch_variants = {
            k: v for k, v in all_directions.items()
            if MODEL_CONFIGS[k]["arch"] == arch
            and k in all_directions
        }
        if len(arch_variants) < 2:
            if arch_variants:
                print(f"\n[CROSS] Only one {arch} variant completed "
                      f"— skipping cross-variant comparison.")
            continue

    
        arch_cfg    = ARCHITECTURE_CONFIGS[arch]
        arch_layers = list(range(arch_cfg["n_layers"]))
        peak_layer  = arch_cfg["peak_layer"]      # 28 for Llama, 40 for Gemma
        peak_layers = arch_cfg["peak_layers"]     # [27-31] or [37-41]

       
        if len(arch_variants) < 2:
            if arch_variants:
                print(f"\n[CROSS] Only one {arch} variant completed "
                      f"— skipping cross-variant comparison.")
            continue

        arch_layers = list(range(
            ARCHITECTURE_CONFIGS[arch]["n_layers"]
        ))

        print(f"\n[CROSS-VARIANT] {arch.upper()} pairwise cosine "
              f"similarities ({len(arch_variants)} variants)...")

        cosine_results = pairwise_cosine_matrix(
            arch_variants, arch_layers
        )

        # Save JSON
        out_path = OUTPUT_DIR / f"cross_variant_cosine_{arch}.json"
        with open(out_path, "w") as f:
            json.dump(
                {str(l): v for l, v in cosine_results.items()},
                f, indent=2,
            )

        # Print summary table — every 4th layer
        variants = list(arch_variants.keys())
        pairs = [
            (variants[i], variants[j])
            for i in range(len(variants))
            for j in range(i + 1, len(variants))
        ]
        header = " | ".join(
            f"{VARIANT_LABELS.get(a,'?')} vs "
            f"{VARIANT_LABELS.get(b,'?')}"
            for a, b in pairs
        )
        print(f"\nLayer | {header}")
        print("-" * (8 + len(header)))
        for l in arch_layers[::4]:
            mat  = np.array(cosine_results[l]["matrix"])
            vl   = cosine_results[l]["variants"]
            vals = " | ".join(
                f"{mat[vl.index(a), vl.index(b)]:>10.4f}"
                for a, b in pairs
            )
            print(f"  {l:02d}  | {vals}")

        # Heatmap — now with correct peak layer annotations
        plot_cosine_heatmap(
            cosine_results, arch_layers, OUTPUT_DIR,
            suffix=arch,
            peak_layers=peak_layers,    # architecture-aware
        )
    
        # Line profiles — now with correct peak layer marker
        plot_cosine_similarity_profiles(
            cosine_results, arch_layers, OUTPUT_DIR,
            suffix=arch,
            peak_layer=peak_layer,      # architecture-aware
        )
    
        # Stability plot unchanged
        arch_stability = {
            k: v for k, v in all_stability.items()
            if MODEL_CONFIGS[k]["arch"] == arch
        }
        plot_stability(
            arch_stability, arch_layers, OUTPUT_DIR, suffix=arch
        )
    # # ── Optional early layer diagnostic ───────────────────────────
    # if RUN_EARLY_LAYER_DIAGNOSTIC and len(all_directions) > 1:
    #     print("\n[EARLY DIAG] Running early layer divergence check...")
    #     print("[EARLY DIAG] Reloading models for diagnostic pass...")
        
    #     # Only run for intra-POST variants of completed models
    #     # Skip base models — diagnostic is about POST variants
    #     diag_models     = {}
    #     diag_tokenizers = {}
        
    #     for name in all_directions.keys():
    #         if MODEL_CONFIGS[name]["is_base"]:
    #             continue
    #         if MODEL_CONFIGS[name]["arch"] != "llama":
    #             continue  # one arch at a time
            
    #         try:
    #             print(f"  Reloading {name}...")
    #             m, tok = load_model(
    #                 MODEL_CONFIGS[name]["path"],
    #                 MODEL_CONFIGS[name]["arch"],
    #             )
    #             diag_models[name]     = m
    #             diag_tokenizers[name] = tok
    #         except Exception as e:
    #             print(f"  [WARN] Could not reload {name}: {e}")
    #             continue
        
    #     if len(diag_models) > 1:
    #         early_diag = check_early_layer_divergence(
    #             models_dict=diag_models,
    #             tokenizers_dict=diag_tokenizers,
    #             harmless_prompts=harmless_prompts,
    #             layers=EARLY_DIAG_LAYERS,
    #         )
            
    #         # Save results
    #         diag_path = OUTPUT_DIR / "early_layer_diagnostic.json"
    #         with open(diag_path, "w") as f:
    #             json.dump(
    #                 {str(l): v for l, v in early_diag.items()},
    #                 f, indent=2,
    #             )
    #         print(f"[EARLY DIAG] Saved → {diag_path}")
            
    #         # Compare against harmful prompt cosine sims
    #         # at same layers from main analysis
    #         print("\n[EARLY DIAG] Comparison:")
    #         print("  Pair | Harmful (Phase1) | Harmless (Diag)")
    #         print("  " + "-" * 50)
            
    #         for name in ["sft_vs_ra_sft",
    #                      "sft_vs_orpo",
    #                      "ra_sft_vs_orpo"]:
    #             n1, n2 = name.split("_vs_")
                
    #             # Get harmful cosine sim at layer 3
    #             # from saved cross_variant_cosine results
    #             cosine_path = (OUTPUT_DIR /
    #                            "cross_variant_cosine_llama.json")
    #             if cosine_path.exists():
    #                 with open(cosine_path) as f:
    #                     cv = json.load(f)
    #                 mat  = np.array(cv["3"]["matrix"])
    #                 vl   = cv["3"]["variants"]
    #                 if n1 in vl and n2 in vl:
    #                     harmful_sim = mat[
    #                         vl.index(n1), vl.index(n2)
    #                     ]
    #                 else:
    #                     harmful_sim = float("nan")
    #             else:
    #                 harmful_sim = float("nan")
                
    #             harmless_sim = early_diag.get(3, {}).get(
    #                 f"{n1}_vs_{n2}",
    #                 early_diag.get(3, {}).get(
    #                     f"{n2}_vs_{n1}", float("nan")
    #                 )
    #             )
                
    #             print(
    #                 f"  {name:20s} | "
    #                 f"{harmful_sim:16.4f} | "
    #                 f"{harmless_sim:.4f}"
    #             )
        
    #     # Free diagnostic models
    #     for m in diag_models.values():
    #         del m
    #     del diag_models, diag_tokenizers
    #     import gc
    #     gc.collect()
    #     torch.cuda.empty_cache()
    
    if RUN_EARLY_LAYER_DIAGNOSTIC:
        print("\n[EARLY DIAG] Running early layer divergence check...")
    
        diag_acts       = {}
        diag_tokenizers = {}
    
        # Load one model at a time, extract, store acts, delete
        for name in all_directions.keys():
            cfg = MODEL_CONFIGS[name]
            if cfg["is_base"] or cfg["arch"] != "llama":
                continue
    
            print(f"[EARLY DIAG] Loading {name}...")
            try:
                m, tok = load_model(cfg["path"], cfg["arch"])
            except Exception as e:
                print(f"[EARLY DIAG] Could not load {name}: {e}")
                continue
    
            # Extract harmless activations at diagnostic layers
            layer_storage = {l: [] for l in EARLY_DIAG_LAYERS}
    
            def make_hook(layer_idx):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output
                    if hidden.ndim == 3:
                        layer_storage[layer_idx].append(
                            hidden[0, -1, :].detach().cpu().float()
                        )
                    else:
                        layer_storage[layer_idx].append(
                            hidden[-1, :].detach().cpu().float()
                        )
                return hook_fn
    
            hooks = [
                m.model.layers[l].register_forward_hook(make_hook(l))
                for l in EARLY_DIAG_LAYERS
            ]
    
            with torch.no_grad():
                for prompt in harmless_prompts[:50]:
                    inputs = tok(
                        format_prompt_native(
                            prompt, cfg["arch"], cfg["is_base"]
                        ),
                        return_tensors="pt",
                    ).to("cuda")
                    m(**inputs)
                    del inputs
                    torch.cuda.empty_cache()
    
            for h in hooks:
                h.remove()
    
            # Store mean acts per layer, then free model immediately
            diag_acts[name] = {
                l: torch.stack(layer_storage[l]).mean(0)
                if layer_storage[l] else None
                for l in EARLY_DIAG_LAYERS
            }
            diag_tokenizers[name] = tok
    
            del m
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[EARLY DIAG] {name} done, model freed.")
    
        # Now compute pairwise cosine similarities from stored acts
        # (no models in memory at this point)
        if len(diag_acts) > 1:
            names = list(diag_acts.keys())
            early_diag = {}
    
            for l in EARLY_DIAG_LAYERS:
                layer_sims = {}
                for i, n1 in enumerate(names):
                    for j, n2 in enumerate(names):
                        if i < j:
                            a1 = diag_acts[n1].get(l)
                            a2 = diag_acts[n2].get(l)
                            if a1 is None or a2 is None:
                                layer_sims[f"{n1}_vs_{n2}"] = float("nan")
                                continue
                            sim = F.cosine_similarity(
                                a1.unsqueeze(0),
                                a2.unsqueeze(0),
                            ).item()
                            layer_sims[f"{n1}_vs_{n2}"] = sim
    
                early_diag[l] = layer_sims
                print(f"\n[EARLY DIAG] Layer {l} "
                      f"cosine sim on harmless prompts:")
                for pair, sim in layer_sims.items():
                    print(f"  {pair}: {sim:.4f}")
    
            # Save
            diag_path = OUTPUT_DIR / "early_layer_diagnostic.json"
            with open(diag_path, "w") as f:
                json.dump(
                    {str(l): v for l, v in early_diag.items()},
                    f, indent=2,
                )
            print(f"[EARLY DIAG] Saved → {diag_path}")
    # ── Magnitude profile plots, separately per architecture ──────────────────
    for arch in ["llama", "gemma", "qwen3"]:
        if not any(
            MODEL_CONFIGS[k]["arch"] == arch
            for k in all_directions
        ):
            continue
    
        arch_layers = list(range(
            ARCHITECTURE_CONFIGS[arch]["n_layers"]
        ))
        base_keys = [
            k for k in all_directions
            if MODEL_CONFIGS[k]["arch"] == arch
            and MODEL_CONFIGS[k]["is_base"]
        ]
    
        # Load normalised and raw profiles from checkpoints
        # (all_mag_profiles only has raw from run_single_model)
        arch_profiles_raw  = {}
        arch_profiles_norm = {}
    
        for k in all_directions:
            if MODEL_CONFIGS[k]["arch"] != arch:
                continue
            ckpt = torch.load(
                OUTPUT_DIR / k / "directions.pt",
                map_location="cpu",
            )
            arch_profiles_raw[k]  = ckpt["magnitude_profile"]
            arch_profiles_norm[k] = ckpt["magnitude_profile_norm"]
    
        if arch_profiles_norm:
            plot_magnitude_profiles(
                arch_profiles_norm, arch_layers,
                OUTPUT_DIR, suffix=arch,
                normalised=True,
                base_keys=base_keys,
            )
        if arch_profiles_raw:
            plot_magnitude_profiles(
                arch_profiles_raw, arch_layers,
                OUTPUT_DIR, suffix=arch,
                normalised=False,
                base_keys=base_keys,
            )

    # Crossover layer report
    if base_keys:
        base_key = base_keys[0]
        base_norm = arch_profiles_norm.get(base_key, {})
        for variant in arch_profiles_norm:
            if variant in base_keys:
                continue
            crossover = find_crossover_layer(
                base_norm,
                arch_profiles_norm[variant],
                arch_layers,
            )
            print(
                f"[CROSSOVER] {arch} {variant} overtakes "
                f"base at layer {crossover}"
            )

    shutdown("Phase 1 complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 1: Refusal direction analysis across "
                    "training objectives"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_CONFIGS.keys()),
        default=None,
        help=(
            "Which variants to run. "
            f"Choices: {list(MODEL_CONFIGS.keys())}. "
            "Defaults to all."
        ),
    )
    args = parser.parse_args()

    try:
        main(model_names=args.models)
    except KeyboardInterrupt:
        shutdown("KeyboardInterrupt")
    except Exception as e:
        import traceback
        traceback.print_exc()
        shutdown(f"Unhandled exception: {e}")