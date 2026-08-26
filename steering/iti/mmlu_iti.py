
#!/usr/bin/env python3
"""
MMLU evaluation for ITI steering on Gemma-2 and Llama-3.x checkpoints.

This is a model-family-aware rewrite of the earlier Gemma-only MMLU ITI script.
It supports:
  - Llama-3.1-8B and Gemma-2 checkpoints
  - separate --tokenizer_path, useful when checkpoints do not save tokenizer files
  - model-family-aware chat templates
  - baseline alpha=0 without registering hooks
  - optional probe_path for baseline-only evaluation
  - robust MMLU loading from JSONL / JSON list / JSON dict
  - generation-based or logits-based multiple-choice scoring

Expected MMLU rows:
  JSONL rows: {"prompt": "...", "answer": "A", "category": "..."}
  JSON array: [{"prompt": "...", "answer": "A"}, ...]

Llama ITI example:
  python mmlu_iti_llama_gemma.py \
    --model_path models/llama-sft \
    --tokenizer_path workspace/tokenizer/llama-3.1-8b-tokenizer \
    --model_name llama-sft \
    --model_family llama \
    --probe_path results/phase3/llama-sft/iti_probes.pt \
    --target_heads 30:25 31:25 \
    --mmlu_path data/mmlu_200.jsonl \
    --alphas 0 5 10 20 \
    --batch_size 4 \
    --max_length 1024 \
    --max_new_tokens 8 \
    --output_dir results/mmlu_iti/llama-sft

Gemma ITI example:
  python mmlu_iti_llama_gemma.py \
    --model_path models/gemma-sft \
    --model_name gemma-sft \
    --model_family gemma \
    --probe_path results/phase3/gemma-sft/iti_probes.pt \
    --target_heads 37:15 38:15 \
    --mmlu_path data/mmlu_200.jsonl \
    --alphas 0 5 10 20 40 \
    --output_dir results/mmlu_iti/gemma-sft

Baseline-only example, no ITI probe needed:
  python mmlu_iti_llama_gemma.py \
    --model_path models/llama-sft \
    --tokenizer_path workspace/tokenizer/llama-3.1-8b-tokenizer \
    --model_name llama-sft \
    --model_family llama \
    --mmlu_path data/mmlu_200.jsonl \
    --alphas 0 \
    --output_dir results/mmlu_baseline/llama-sft
"""

from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

TargetHead = Tuple[int, int]
DEFAULT_SYSTEM_PROMPT = "You are a helpful, harmless, and honest assistant."
ANSWER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def parse_target_heads(items: Optional[Sequence[str]]) -> List[TargetHead]:
    if not items:
        return []
    out: List[TargetHead] = []
    for item in items:
        try:
            layer_s, head_s = item.split(":")
            out.append((int(layer_s), int(head_s)))
        except Exception as exc:
            raise argparse.ArgumentTypeError(
                f"Bad target head {item!r}; expected LAYER:HEAD, e.g. 30:25"
            ) from exc
    return out


def infer_model_family(model_path: str, model_name: str, explicit: str = "auto") -> str:
    if explicit and explicit.lower() != "auto":
        family = explicit.lower()
    else:
        text = f"{model_path} {model_name}".lower()
        if "llama" in text:
            family = "llama"
        elif "gemma" in text:
            family = "gemma"
        else:
            family = "unknown"
    if family not in {"llama", "gemma", "unknown"}:
        raise ValueError("--model_family must be one of: auto, llama, gemma, unknown")
    return family


def get_input_device(model) -> torch.device:
    """Return a safe device for input tensors under either normal or device_map loading."""
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        for _, dev in model.hf_device_map.items():
            if isinstance(dev, str) and dev not in {"cpu", "disk"}:
                return torch.device(dev)
    return next(model.parameters()).device


def json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        if obj.numel() <= 20:
            return obj.detach().cpu().tolist()
        return {
            "__tensor__": True,
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "norm": float(obj.detach().float().norm().cpu()),
        }
    if isinstance(obj, (torch.dtype, torch.device)):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(obj), indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _row_to_sample(row: Dict[str, Any], i: int) -> Dict[str, str]:
    prompt = row.get("prompt") or row.get("question") or row.get("input")
    answer = row.get("answer") or row.get("target") or row.get("label")
    if prompt is None or answer is None:
        raise KeyError(f"MMLU row {i} missing prompt/answer keys: {list(row.keys())}")

    answer_s = str(answer).strip().upper()
    if answer_s in {"0", "1", "2", "3"}:
        answer_s = "ABCD"[int(answer_s)]
    answer_s = answer_s[0]
    if answer_s not in "ABCD":
        raise ValueError(f"MMLU row {i} has invalid answer={answer!r}")

    return {
        "prompt": str(prompt),
        "answer": answer_s,
        "category": str(row.get("category", row.get("subject", "unknown"))),
    }


def load_mmlu(path: Path, limit: Optional[int] = None) -> List[Dict[str, str]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            raw_rows = obj
        elif isinstance(obj, dict):
            if "samples" in obj:
                raw_rows = obj["samples"]
            elif "data" in obj:
                raw_rows = obj["data"]
            else:
                raw_rows = [obj]
        else:
            raise TypeError(f"Unsupported JSON root type: {type(obj)}")
    except Exception:
        raw_rows = [json.loads(line) for line in text.splitlines() if line.strip()]

    samples: List[Dict[str, str]] = []
    for i, row in enumerate(raw_rows):
        if not isinstance(row, dict):
            raise TypeError(f"MMLU row {i} is not a dict: {type(row)}")
        samples.append(_row_to_sample(row, i))
        if limit is not None and len(samples) >= limit:
            break

    print(f"Loaded {len(samples)} MMLU samples from {path}")
    return samples


# ---------------------------------------------------------------------------
# Model, tokenizer, and prompt formatting
# ---------------------------------------------------------------------------

def load_tokenizer(tokenizer_path: str):
    tok = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return tok


def load_model(model_path: str, dtype: torch.dtype, device_map: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    return model


def format_mmlu_prompt(
    prompt: str,
    tokenizer,
    model_family: str,
    use_chat_template: bool,
    system_prompt: str,
) -> str:
    instruction = f"{prompt}\n\nAnswer with only one letter: A, B, C, or D."

    if not use_chat_template:
        return instruction
    if tokenizer.chat_template is None:
        return instruction

    try:
        if model_family == "gemma":
            # Gemma chat templates commonly do not support a separate system role.
            messages = [{"role": "user", "content": f"{system_prompt}\n\n{instruction}"}]
        elif model_family == "llama":
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction},
            ]
        else:
            # Conservative fallback for custom chat templates.
            messages = [{"role": "user", "content": f"{system_prompt}\n\n{instruction}"}]

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return instruction


# ---------------------------------------------------------------------------
# ITI hooks
# ---------------------------------------------------------------------------

def get_num_heads_and_head_dim(model) -> Tuple[int, int]:
    cfg = model.config
    n_heads = int(getattr(cfg, "num_attention_heads"))
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        head_dim = int(getattr(cfg, "hidden_size")) // n_heads
    return n_heads, int(head_dim)


def prepare_probe_obj_for_model(model, probe_obj: Dict[str, object], target_heads: Sequence[TargetHead]) -> Dict[str, object]:
    if not target_heads:
        return probe_obj
    if "probes" not in probe_obj:
        raise KeyError("probe_path object must contain a 'probes' dictionary")

    probes: Dict[str, Tensor] = probe_obj["probes"]
    prepared = dict(probe_obj)
    prepared_probes: Dict[str, Tensor] = dict(probes)

    n_heads, _ = get_num_heads_and_head_dim(model)
    n_layers = len(model.model.layers)

    for layer_idx, head_idx in target_heads:
        key = f"{layer_idx}:{head_idx}"
        if layer_idx < 0 or layer_idx >= n_layers:
            raise ValueError(f"layer_idx={layer_idx} out of range for model with {n_layers} layers")
        if head_idx < 0 or head_idx >= n_heads:
            raise ValueError(f"head_idx={head_idx} out of range for model with {n_heads} attention heads")
        if key not in probes:
            raise KeyError(f"Missing ITI probe {key}; available keys include {list(probes.keys())[:20]}")

        layer_device = next(model.model.layers[layer_idx].parameters()).device
        prepared_probes[key] = probes[key].detach().float().to(device=layer_device)

    prepared["probes"] = prepared_probes
    return prepared


def make_iti_pre_hook(probe_dir: Tensor, alpha: float, head_idx: int, head_dim: int, pos: int):
    start = head_idx * head_dim
    end = (head_idx + 1) * head_dim

    def pre_hook(module, inputs):
        x = inputs[0]
        seq_len = x.shape[1]
        real_pos = pos if pos >= 0 else seq_len + pos
        if real_pos < 0 or real_pos >= seq_len:
            return inputs

        x2 = x.clone()
        d = probe_dir.to(device=x2.device, dtype=x2.dtype)
        x2[:, real_pos, start:end] += alpha * d
        return (x2,) + tuple(inputs[1:])

    return pre_hook


def register_iti_hooks(model, probe_obj: Dict[str, object], target_heads: Sequence[TargetHead], alpha: float, pos: int):
    _, head_dim = get_num_heads_and_head_dim(model)
    probes: Dict[str, Tensor] = probe_obj["probes"]
    handles = []
    for layer_idx, head_idx in target_heads:
        key = f"{layer_idx}:{head_idx}"
        o_proj = model.model.layers[layer_idx].self_attn.o_proj
        handles.append(o_proj.register_forward_pre_hook(make_iti_pre_hook(probes[key], alpha, head_idx, head_dim, pos)))
    return handles


# ---------------------------------------------------------------------------
# Answer extraction / scoring
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> str:
    s = text.strip()
    if not s:
        return ""

    # Common exact outputs: A, A., A), "A\n".
    first = re.match(r"^\s*([ABCD])(?:\b|[\).:,-])", s, flags=re.IGNORECASE)
    if first:
        return first.group(1).upper()

    # Common verbose outputs: "The answer is A".
    for pattern in [
        r"answer\s+is\s+([ABCD])\b",
        r"option\s+([ABCD])\b",
        r"choice\s+([ABCD])\b",
        r"\(([ABCD])\)",
    ]:
        m = re.search(pattern, s, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

    m = ANSWER_RE.search(s)
    return m.group(1).upper() if m else ""


def answer_token_ids(tokenizer, device: torch.device) -> Dict[str, Tensor]:
    """
    Candidate token IDs for A-D. Handles tokenizers where 'A' and ' A'
    are different single-token encodings. Multi-token candidates are ignored.
    """
    out: Dict[str, List[int]] = {k: [] for k in "ABCD"}
    for letter in "ABCD":
        candidates = [letter, f" {letter}", f"\n{letter}"]
        for cand in candidates:
            ids = tokenizer.encode(cand, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in out[letter]:
                out[letter].append(ids[0])
        if not out[letter]:
            raise ValueError(f"Could not find a single-token candidate for answer {letter!r}")
    return {k: torch.tensor(v, device=device, dtype=torch.long) for k, v in out.items()}


@torch.no_grad()
def score_logits(model, tokenizer, inputs, answer_ids: Dict[str, Tensor]) -> List[str]:
    outputs = model(**inputs, use_cache=False)
    logits = outputs.logits[:, -1, :]  # next-token logits after the prompt
    preds: List[str] = []
    for b in range(logits.shape[0]):
        scores = {}
        for letter, ids in answer_ids.items():
            scores[letter] = float(logits[b, ids].max().detach().cpu())
        preds.append(max(scores.items(), key=lambda kv: kv[1])[0])
    return preds


@torch.no_grad()
def score_generate(model, tokenizer, inputs, max_new_tokens: int) -> Tuple[List[str], List[str]]:
    input_len = inputs["input_ids"].shape[1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
    preds = [extract_answer(r) for r in decoded]
    return preds, [r.strip() for r in decoded]


@torch.no_grad()
def evaluate_mmlu_for_alpha(
    model,
    tokenizer,
    samples: Sequence[Dict[str, str]],
    probe_obj: Optional[Dict[str, object]],
    target_heads: Sequence[TargetHead],
    alpha: float,
    pos: int,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
    model_family: str,
    use_chat_template: bool,
    system_prompt: str,
    save_examples: int,
    eval_mode: str,
) -> Dict[str, object]:
    correct = 0
    total = 0
    examples: List[Dict[str, Any]] = []
    by_category: Dict[str, Dict[str, int]] = {}
    input_device = get_input_device(model)
    ans_ids = answer_token_ids(tokenizer, input_device) if eval_mode == "logits" else None

    for start in tqdm(range(0, len(samples), batch_size), desc=f"MMLU alpha={alpha}"):
        batch_samples = list(samples[start:start + batch_size])
        texts = [
            format_mmlu_prompt(s["prompt"], tokenizer, model_family, use_chat_template, system_prompt)
            for s in batch_samples
        ]

        handles = []
        try:
            if alpha != 0:
                if probe_obj is None or not target_heads:
                    raise ValueError("Non-zero alpha requires --probe_path and --target_heads")
                handles = register_iti_hooks(model, probe_obj, target_heads, alpha, pos)

            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            ).to(input_device)

            if eval_mode == "logits":
                predictions = score_logits(model, tokenizer, inputs, ans_ids)  # type: ignore[arg-type]
                responses = [""] * len(predictions)
            else:
                predictions, responses = score_generate(model, tokenizer, inputs, max_new_tokens)

        finally:
            for h in handles:
                h.remove()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        for sample, pred, response in zip(batch_samples, predictions, responses):
            gold = sample["answer"]
            ok = pred == gold
            correct += int(ok)
            total += 1
            cat = sample.get("category", "unknown")
            by_category.setdefault(cat, {"correct": 0, "total": 0})
            by_category[cat]["correct"] += int(ok)
            by_category[cat]["total"] += 1

            if len(examples) < save_examples:
                examples.append({
                    "prompt": sample["prompt"],
                    "answer": gold,
                    "prediction": pred,
                    "response": response,
                    "correct": ok,
                    "category": cat,
                })

    category_metrics = {
        cat: {
            "accuracy": round(v["correct"] / max(1, v["total"]), 4),
            "correct": v["correct"],
            "total": v["total"],
        }
        for cat, v in sorted(by_category.items())
    }
    acc = correct / max(1, total)
    return {
        "accuracy": round(acc, 4),
        "correct": correct,
        "total": total,
        "by_category": category_metrics,
        "examples": examples,
    }


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="MMLU utility evaluation with ITI hooks for Llama/Gemma")

    p.add_argument("--model_path", required=True, help="Path or HF id for the evaluated model checkpoint")
    p.add_argument("--tokenizer_path", default=None, help="Optional separate tokenizer path. Defaults to --model_path")
    p.add_argument("--model_name", required=True)
    p.add_argument("--model_family", choices=["auto", "llama", "gemma", "unknown"], default="auto")

    p.add_argument("--probe_path", type=Path, default=None, help="ITI probe .pt file. Required when any alpha is non-zero")
    p.add_argument("--target_heads", nargs="*", default=[], help="Example for Llama: 30:25 31:25")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0])
    p.add_argument("--pos", type=int, default=-1)

    p.add_argument("--mmlu_path", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)

    p.add_argument("--eval_mode", choices=["generate", "logits"], default="generate",
                   help="generate matches your previous script; logits scores A/B/C/D next-token likelihood directly")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--device_map", default="auto")

    p.add_argument("--output_dir", type=Path, default=Path("results/mmlu_iti"))
    p.add_argument("--save_examples", type=int, default=10)
    p.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    p.add_argument("--no_chat_template", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_family = infer_model_family(args.model_path, args.model_name, args.model_family)
    tokenizer_path = args.tokenizer_path or args.model_path
    target_heads = parse_target_heads(args.target_heads)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    if any(alpha != 0 for alpha in args.alphas):
        if args.probe_path is None:
            raise ValueError("At least one alpha is non-zero, so --probe_path is required")
        if not target_heads:
            raise ValueError("At least one alpha is non-zero, so --target_heads is required")

    samples = load_mmlu(args.mmlu_path, args.limit)

    print(f"Model family: {model_family}")
    print(f"Loading tokenizer: {tokenizer_path}")
    tokenizer = load_tokenizer(tokenizer_path)

    print(f"Loading model: {args.model_path}")
    model = load_model(args.model_path, dtype=dtype, device_map=args.device_map)

    probe_obj: Optional[Dict[str, object]] = None
    if args.probe_path is not None:
        print(f"Loading ITI probes: {args.probe_path}")
        probe_obj = torch.load(args.probe_path, map_location="cpu")
        probe_obj = prepare_probe_obj_for_model(model, probe_obj, target_heads)

    results: Dict[str, Any] = {
        "method": "iti" if args.probe_path is not None else "baseline",
        "metric": "mmlu_accuracy",
        "eval_mode": args.eval_mode,
        "model_name": args.model_name,
        "model_path": args.model_path,
        "tokenizer_path": tokenizer_path,
        "model_family": model_family,
        "probe_path": str(args.probe_path) if args.probe_path is not None else None,
        "target_heads": [f"{l}:{h}" for l, h in target_heads],
        "mmlu_path": str(args.mmlu_path),
        "limit": args.limit,
        "pos": args.pos,
        "alphas": args.alphas,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "use_chat_template": not args.no_chat_template,
        "results": {},
    }

    for alpha in args.alphas:
        metrics = evaluate_mmlu_for_alpha(
            model=model,
            tokenizer=tokenizer,
            samples=samples,
            probe_obj=probe_obj,
            target_heads=target_heads,
            alpha=alpha,
            pos=args.pos,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            model_family=model_family,
            use_chat_template=not args.no_chat_template,
            system_prompt=args.system_prompt,
            save_examples=args.save_examples,
            eval_mode=args.eval_mode,
        )
        results["results"][str(alpha)] = metrics
        print(f"alpha={alpha}: accuracy={metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")

        ckpt_path = args.output_dir / f"{args.model_name}_mmlu_{args.eval_mode}_ckpt.json"
        write_json(ckpt_path, results)

    alpha_tag = "_".join(str(a).replace("-", "neg").replace(".", "p") for a in args.alphas)
    suffix = "iti" if args.probe_path is not None else "baseline"
    out_path = args.output_dir / f"{args.model_name}_mmlu_{suffix}_{args.eval_mode}_alphas_{alpha_tag}.json"
    write_json(out_path, results)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
