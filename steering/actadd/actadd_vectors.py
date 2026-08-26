import torch
import json
import argparse
from pathlib import Path
from typing import List, Optional
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re


def get_stop_token_ids(tokenizer, arch: Optional[str] = None) -> List[int]:
    """
    Resolve the actual stopping token(s) for generate(), rather than
    trusting tokenizer.eos_token_id (or model.config.eos_token_id) alone.

    Qwen3 has a documented tokenizer/template mismatch: a recent tokenizer
    update changed eos_token from "<|im_end|>" to "<|endoftext|>" to match
    base-model behaviour, but the chat template still ends every assistant
    turn with "<|im_end|>" and never emits "<|endoftext|>". A fine-tuned
    Qwen3 checkpoint correctly learns to emit "<|im_end|>" at the end of a
    response -- but if generate() only watches for eos_token_id (now
    "<|endoftext|>"), it never recognizes that signal, so generation runs
    to max_new_tokens every time. See QwenLM/Qwen3 issues #927, #1064.

    Detection is via direct vocab membership, not an arch-name string
    match -- gating this behind arch=="qwen3" silently does nothing if
    --arch is left at its default (None) or doesn't match exactly.
    Checking the vocab directly is a safe no-op for Llama/Gemma.
    """
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)

    vocab = tokenizer.get_vocab()
    if "<|im_end|>" in vocab:
        stop_ids.add(vocab["<|im_end|>"])

    return sorted(stop_ids)


# ─────────────────────────────────────────────────────────────────────────────
# Direction loading
# ─────────────────────────────────────────────────────────────────────────────
def load_refusal_direction(
    hf_repo: str,
    filename: str,
    peak_layer: int,
    device: str,
):
    print(f"Loading direction from {hf_repo} / {filename}")

    local_path = hf_hub_download(
        repo_id=hf_repo,
        filename=filename,
        repo_type="dataset",
        token=True,
    )

    d = torch.load(local_path, map_location=device)

    if peak_layer not in d["layers"]:
        raise ValueError(
            f"peak_layer={peak_layer} not in stored layers {d['layers']}"
        )

    direction = d["directions"][peak_layer].float()
    direction = direction / direction.norm()

    return direction, d


# ─────────────────────────────────────────────────────────────────────────────
# Save steering vector
# ─────────────────────────────────────────────────────────────────────────────

def save_steering_vector(
    direction: torch.Tensor,
    metadata: dict,
    output_dir: Path,
    model_name: str,
    peak_layer: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / f"{model_name}_layer{peak_layer}_vector.pt"

    save_obj = {
        "direction": direction.cpu(),
        "metadata": metadata,
    }

    torch.save(save_obj, save_path)

    print(f"Saved steering vector → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Activation Addition Hook
# ─────────────────────────────────────────────────────────────────────────────

def make_actadd_hook(direction: torch.Tensor, alpha: float, pos: int):
    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output

        hidden[:, pos, :] = hidden[:, pos, :] + alpha * direction

        if isinstance(output, tuple):
            return (hidden,) + output[1:]

        return hidden

    return hook


# ─────────────────────────────────────────────────────────────────────────────
# Steered generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_steered(
    model,
    tokenizer,
    input_ids,
    direction,
    alpha,
    target_layers,
    pos=-1,
    max_new_tokens=256,
    arch=None,
):
    handles = []

    for layer_idx in target_layers:
        layer = model.model.layers[layer_idx]

        handles.append(
            layer.register_forward_hook(
                make_actadd_hook(direction, alpha, pos)
            )
        )

    stop_token_ids = get_stop_token_ids(tokenizer, arch)

    try:
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=stop_token_ids,
            )

    finally:
        for h in handles:
            h.remove()

    return output[0][input_ids.shape[1]:]


# ─────────────────────────────────────────────────────────────────────────────
# MMLU Evaluation
# ─────────────────────────────────────────────────────────────────────────────

# SPECIAL_STRINGS = [
#     "<END_OF_TURN>",
#     "<START_OF_TURN>",
#     "MODEL",
# ]

# def extract_answer(text):
#     t = text.upper()
#     t = t.replace("<END_OF_TURN>", " ").replace("<START_OF_TURN>", " ")

#     patterns = [
#         r"THE\s+CORRECT\s+ANSWER\s+IS\s+([ABCD])",
#         r"CORRECT\s+ANSWER\s+IS\s+([ABCD])",
#         r"FINAL\s+ANSWER\s*[:\-]?\s*([ABCD])",
#         r"ANSWER\s*[:\-]?\s*([ABCD])",
#         r"CORRESPONDS\s+TO\s+OPTION\s+([ABCD])",
#         r"OPTION\s+([ABCD])",
#         r"CHOICE\s+([ABCD])",
#         r"^\s*([ABCD])\b",
#     ]

    
#     for p in patterns:
#         m = re.search(p, t)
#         if m:
#             return m.group(1)
#     return ""

def remove_think_blocks(text):
    return re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    
def extract_answer(text):
    # BUG FIX: this previously computed remove_think_blocks(text) and then
    # immediately discarded it by reassigning from the raw `text` on the
    # next line -- meaning think-block removal never actually happened.
    # For any Ra-SFT checkpoint (Llama, Gemma, or Qwen3), the <think>
    # block can discuss multiple answer options before converging on a
    # final one, so the regex patterns below could match an option letter
    # mentioned mid-reasoning rather than the model's actual final answer.
    t = remove_think_blocks(text).upper()

    # Remove obvious chat artifacts only
    for s in [
        "<END_OF_TURN>", "<START_OF_TURN>",
        "<|BEGIN_OF_TEXT|>", "<|START_HEADER_ID|>",
        "<|END_HEADER_ID|>", "<|EOT_ID|>",
        "<|IM_START|>", "<|IM_END|>",  # Qwen3
        "MODEL", "ASSISTANT"
    ]:
        t = t.replace(s, " ")

    # Best explicit patterns first
    patterns = [
        r"THE\s+CORRECT\s+ANSWER\s+IS\s+([ABCD])",
        r"CORRECT\s+ANSWER\s+IS\s+([ABCD])",
        r"ANSWER\s+IS\s+([ABCD])",
        r"ANSWER\s*[:\-]\s*([ABCD])",
        r"OPTION\s+([ABCD])",
        r"CHOICE\s+([ABCD])",
        r"^\s*([ABCD])\b",
    ]

    for p in patterns:
        m = re.search(p, t)
        if m:
            return m.group(1)

    # old-style fallback: first standalone A/B/C/D
    m = re.search(r"\b([ABCD])\b", t)
    return m.group(1) if m else ""
    
def is_ra_sft(model_name: str) -> bool:
    return "ra-sft" in model_name.lower()


def format_mmlu_prompt(prompt, arch, tokenizer, model_name):
    """
    arch=None (or anything other than "qwen3") falls back to the
    original generic apply_chat_template call -- unchanged behaviour
    for Llama/Gemma. Qwen3 gets explicit paradigm-aware formatting for
    the same reason as every other Qwen3 eval script in this pipeline:
    apply_chat_template's default enable_thinking=True doesn't match
    how the SFT/ORPO checkpoints were trained, and silently accepting
    that default would mean this steering-vs-capability measurement is
    partly measuring "prompt format is unfamiliar" rather than purely
    "steering degraded capability."
    """
    if arch == "qwen3":
        if is_ra_sft(model_name):
            messages = [{"role": "user", "content": prompt}]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def evaluate_mmlu(
    model,
    tokenizer,
    mmlu_path: Path,
    direction,
    target_layers,
    alpha,
    pos,
    device,
    arch=None,
    model_name="",
):
    samples = []

    with open(mmlu_path) as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    correct = 0

    for sample in tqdm(samples, desc=f"MMLU α={alpha}"):
        formatted = format_mmlu_prompt(
            sample["prompt"], arch, tokenizer, model_name
        )
        
        inputs = tokenizer(
            formatted,
            return_tensors="pt"
        ).to(device)

        new_tokens = generate_steered(
            model=model,
            tokenizer=tokenizer,
            input_ids=inputs["input_ids"],
            direction=direction,
            alpha=alpha,
            target_layers=target_layers,
            pos=pos,
            max_new_tokens=256,
            arch=arch,
        )

        decoded = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

        predicted = extract_answer(decoded)

        if predicted == sample["answer"]:
            correct += 1

    acc = round(correct / len(samples), 4)

    return acc


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_mmlu_evaluation(
    model,
    tokenizer,
    mmlu_path,
    direction,
    target_layers,
    alphas,
    pos,
    model_name,
    output_dir,
    device,
    arch=None,
):
    results = {
        "model_name": model_name,
        "target_layers": target_layers,
        "alphas": alphas,
        "results": {},
    }

    for alpha in alphas:

        print(f"\nRunning MMLU with α={alpha}")

        mmlu_acc = evaluate_mmlu(
            model=model,
            tokenizer=tokenizer,
            mmlu_path=mmlu_path,
            direction=direction,
            target_layers=target_layers,
            alpha=alpha,
            pos=pos,
            device=device,
            arch=arch,
            model_name=model_name,
        )

        results["results"][str(alpha)] = {
            "mmlu_accuracy": mmlu_acc
        }

        print(f"MMLU Accuracy: {mmlu_acc:.4f}")

        # checkpoint save every alpha
        output_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = output_dir / f"{model_name}_mmlu_ckpt.json"

        with open(ckpt_path, "w") as f:
            json.dump(results, f, indent=2)

    final_path = output_dir / f"{model_name}_mmlu_results.json"

    with open(final_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved final results → {final_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():

    parser = argparse.ArgumentParser(
        description="Activation Addition MMLU Evaluation"
    )

    # model
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--arch", type=str, default=None,
        choices=[None, "llama", "gemma", "qwen3"],
        help=(
            "Only affects prompt formatting. None (default) uses the "
            "generic apply_chat_template call, unchanged for Llama/Gemma. "
            "qwen3 enables paradigm-aware formatting matching training "
            "format (system message + enable_thinking=False for SFT/ORPO, "
            "no system message + enable_thinking=True for Ra-SFT)."
        ),
    )

    # direction
    parser.add_argument("--direction_repo", type=str, required=True)
    parser.add_argument("--direction_filename", type=str, required=True)
    parser.add_argument("--peak_layer", type=int, required=True)

    # steering
    parser.add_argument(
        "--target_layers",
        type=int,
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.0, -5.0, -10.0, -15.0, -20.0],
    )

    parser.add_argument(
        "--pos",
        type=int,
        default=-1,
    )

    # mmlu
    parser.add_argument(
        "--mmlu_prompts",
        type=Path,
        required=True,
    )

    # output
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/mmlu"),
    )

    # hardware
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():

    args = parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    dtype = dtype_map[args.dtype]

    # load model
    print(f"Loading model: {args.model_path}")
    print(f"Loading tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=args.device,
        trust_remote_code=True,
    )

    model.eval()

    # load direction
    direction, metadata = load_refusal_direction(
        args.direction_repo,
        args.direction_filename,
        args.peak_layer,
        args.device,
    )

    # save vector immediately
    save_steering_vector(
        direction=direction,
        metadata=metadata,
        output_dir=args.output_dir / "vectors",
        model_name=args.model_name,
        peak_layer=args.peak_layer,
    )

    # run MMLU
    run_mmlu_evaluation(
        model=model,
        tokenizer=tokenizer,
        mmlu_path=args.mmlu_prompts,
        direction=direction,
        target_layers=args.target_layers,
        alphas=args.alphas,
        pos=args.pos,
        model_name=args.model_name,
        output_dir=args.output_dir,
        device=args.device,
        arch=args.arch,
    )


if __name__ == "__main__":
    main()