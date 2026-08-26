"""
WildJailbreak adversarial harmful evaluation — all checkpoints sequential.
Judge: LlamaGuard-3 (meta-llama/Llama-Guard-3-8B)
Two-phase: generate all → judge all (judge loaded once).

Pass --skip_generation to re-judge already-cached generations (from a
previous run's generation_cache/{name}.json files) without regenerating
and without loading any generation models — only the judge is loaded.
Use --models to restrict this to specific cached checkpoints, and
--judge_revision to pin a specific judge commit for reproducibility.
This folds in what used to be the separate rejudge_cached_responses.py,
so a rejudge no longer means loading LlamaGuard-3-8B in one script for
the normal run and again in a second script to re-score cached output.
"""

import torch
import json
import argparse
from pathlib import Path
from typing import List, Optional
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import os


def get_stop_token_ids(tokenizer, family: Optional[str] = None) -> List[int]:
    """
    Resolve the actual stopping token(s) for generate(), rather than
    trusting tokenizer.eos_token_id alone.

    Qwen3 has a documented tokenizer/template mismatch: a recent tokenizer
    update changed eos_token from "<|im_end|>" to "<|endoftext|>" to match
    base-model behaviour, but the chat template still ends every assistant
    turn with "<|im_end|>" and never emits "<|endoftext|>". A fine-tuned
    Qwen3 checkpoint correctly learns to emit "<|im_end|>" at the end of a
    response -- but if generate() only watches for eos_token_id (now
    "<|endoftext|>"), it never recognizes that signal, so generation runs
    to max_new_tokens every time. See QwenLM/Qwen3 issues #927, #1064.

    Detection is via direct vocab membership, not a family-name string
    match -- gating this behind family=="qwen3" silently does nothing if
    the caller passes a family string that doesn't happen to match.
    Checking the vocab directly is a safe no-op for Llama/Gemma.
    """
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)

    vocab = tokenizer.get_vocab()
    if "<|im_end|>" in vocab:
        stop_ids.add(vocab["<|im_end|>"])

    return sorted(stop_ids)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge_path", type=str,
                        default="meta-llama/Llama-Guard-3-8B")
    # Pin a specific revision if you have the hash from the original run —
    # from_pretrained without `revision` tracks the repo's current default,
    # which can silently change between pulls if HF updates it upstream.
    parser.add_argument("--judge_revision", type=str, default=None,
                        help="Optional commit hash to pin the judge model, "
                             "for reproducibility across runs")
    parser.add_argument("--judge_max_length", type=int, default=4096,
                        help="Tokenizer truncation length for the judge "
                             "prompt (prompt+response). 4096 avoids "
                             "silently truncating long responses, which "
                             "can shift ASR.")
    parser.add_argument("--output_path", type=str,
                        default="wildjailbreak_results.json")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--cache_dir", type=str, default="generation_cache",
                        help="Directory for per-model response caches")
    parser.add_argument("--skip_generation", action="store_true",
                        help="Re-judge mode: skip Phase 1 entirely (no "
                             "generation models loaded) and re-run "
                             "LlamaGuard classification on the existing "
                             "per-model caches in --cache_dir")
    parser.add_argument("--models", type=str, nargs="*", default=None,
                        help="With --skip_generation, which cached model "
                             "names to re-judge (matching cache filenames "
                             "without .json). Default: all *.json in "
                             "cache_dir except tactics.json")
    return parser.parse_args()


def load_tokenizer(path, revision=None):
    """
    Load tokenizer and ensure pad token is set.
    Llama and LlamaGuard have no pad token by default —
    using eos_token is consistent with how TRL fine-tuned these models.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        path, revision=revision, trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer
    

# ── Model registry ────────────────────────────────────────────────────────────
# Edit paths to match your HuggingFace repo names or local paths.
# Add Gemma configs here when the second instance is ready.
MODEL_CONFIGS = [
    {
        "name": "llama_base",
        "path": "models/llama-base",
        "family": "llama"
    }
    # {
    #     "name": "llama_ra_sft",
    #     "path": "models/llama-ra-sft",
    #     "family": "llama"
    # },
    # {
    #     "name": "llama_orpo",
    #     "path": "models/llama-orpo",
    #     "family": "llama"
    # },
    # Uncomment when Gemma instance is ready:
    # {
    #     "name": "gemma_sft",
    #     "path": "your-hf-username/gemma-safety-sft",
    #     "family": "gemma"
    # },
    # {
    #     "name": "gemma_orpo",
    #     "path": "your-hf-username/gemma-safety-orpo",
    #     "family": "gemma"
    # },
    # Qwen3 -- "family" must be "qwen3" (not "qwen") to match format_prompt's
    # dispatch below. is_ra_sft controls enable_thinking at eval time --
    # see format_prompt for why this isn't optional cosmetic detail.
    # {
    #     "name": "qwen3_sft",
    #     "path": "your-hf-username/qwen3-safety-sft",
    #     "family": "qwen3",
    #     "is_ra_sft": False,
    # },
    # {
    #     "name": "qwen3_ra_sft",
    #     "path": "your-hf-username/qwen3-safety-ra-sft",
    #     "family": "qwen3",
    #     "is_ra_sft": True,
    # },
    # {
    #     "name": "qwen3_orpo",
    #     "path": "your-hf-username/qwen3-safety-orpo",
    #     "family": "qwen3",
    #     "is_ra_sft": False,
    # },
]


# ── Data loading ──────────────────────────────────────────────────────────────
def load_adversarial_harmful(max_samples=None):
    dataset = load_dataset(
        "allenai/wildjailbreak",
        "eval",
        delimiter="\t",
        keep_default_na=False,
        split="train"
    )
    adversarial_harmful = dataset.filter(
        lambda x: x["data_type"] == "adversarial_harmful"
    )
    # Add temporarily after the filter line, before the max_samples check
    print("Column names:", adversarial_harmful.column_names)
    print("First example:", adversarial_harmful[0])
    if max_samples is not None:
        adversarial_harmful = adversarial_harmful.select(
            range(min(max_samples, len(adversarial_harmful)))
        )
    print(f"Loaded {len(adversarial_harmful)} adversarial harmful examples")
    

    return adversarial_harmful


# ── Prompt formatting ─────────────────────────────────────────────────────────
def format_prompt(prompt, model_family, tokenizer, is_ra_sft=False):
    # Qwen3 is handled explicitly, not via the generic try/except below --
    # apply_chat_template's default enable_thinking=True doesn't match how
    # these checkpoints were trained (SFT/ORPO used enable_thinking=False;
    # Ra-SFT used no forced stub at all), and the try/except would silently
    # accept the wrong default rather than erroring, which is worse than
    # a crash here.
    if model_family == "qwen3":
        messages = [{"role": "user", "content": prompt}]
        if is_ra_sft:
            # No system message, enable_thinking=True (no stub forced) --
            # matches train_sft_cot_qwen3.py's hand-rolled format.
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
        # SFT/ORPO: system message + enable_thinking=False, matching
        # train_sft_qwen3.py / train_orpo_qwen3.py exactly.
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )

    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        pass

    if model_family == "llama":
        return (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    elif model_family == "gemma":
        return (
            "<bos>"
            "<start_of_turn>user\n"
            f"{prompt}"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
    else:
        raise ValueError(f"Unknown model family: {model_family}")


def format_judge_prompt(prompt, response, judge_tokenizer):
    """
    LlamaGuard-3 is always Llama-based regardless of target model family.
    apply_chat_template on the judge tokenizer handles the safety
    classification format automatically.
    """
    try:
        return judge_tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ],
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        # Explicit LlamaGuard-3 fallback
        return (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{response}"
            "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )


# ── Generation ────────────────────────────────────────────────────────────────
def generate_responses(model, tokenizer, prompts, batch_size,
                       max_new_tokens, model_family, is_ra_sft=False):
    responses = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i: i + batch_size]
        formatted = [
            format_prompt(p, model_family, tokenizer, is_ra_sft=is_ra_sft)
            for p in batch
        ]
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(model.device)

        stop_token_ids = get_stop_token_ids(tokenizer, model_family)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=stop_token_ids,
            )

        input_len = inputs["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(
            outputs[:, input_len:],
            skip_special_tokens=True
        )
        responses.extend(decoded)

        if (i // batch_size) % 10 == 0:
            print(f"  Generated {i + len(batch)}/{len(prompts)}")
    return responses


# ── LlamaGuard-3 judging ──────────────────────────────────────────────────────
def llamaguard_classify(judge_model, judge_tokenizer, prompts,
                        responses, batch_size, max_length=4096):
    labels = []
    raw_labels = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i: i + batch_size]
        batch_responses = responses[i: i + batch_size]

        formatted = [
            format_judge_prompt(p, r, judge_tokenizer)
            for p, r in zip(batch_prompts, batch_responses)
        ]
        inputs = judge_tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(judge_model.device)

        with torch.no_grad():
            outputs = judge_model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=judge_tokenizer.eos_token_id
            )

        input_len = inputs["input_ids"].shape[1]
        decoded = judge_tokenizer.batch_decode(
            outputs[:, input_len:],
            skip_special_tokens=True
        )
        for label in decoded:
            labels.append("unsafe" in label.strip().lower())
            raw_labels.append(label.strip())

        if (i // batch_size) % 10 == 0:
            print(f"  Judged {i + len(batch_prompts)}/{len(prompts)}")
    return labels, raw_labels


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(exist_ok=True)

    # generation_cache maps name -> {"model", "family", "prompts", "responses"}.
    # Populated either by Phase 1 (generation, below) or by loading existing
    # per-model caches directly when --skip_generation is set. Either way,
    # Phase 2 (judging) only ever reads from this dict, so the two modes
    # share one judging code path and one judge model load.
    generation_cache = {}

    if args.skip_generation:
        # ── Re-judge mode: load existing caches, no generation models ────────
        if args.models:
            cache_paths = [cache_dir / f"{name}.json" for name in args.models]
        else:
            cache_paths = sorted(
                p for p in cache_dir.glob("*.json") if p.stem != "tactics"
            )
        missing = [p for p in cache_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Cache files not found: {missing}")

        print(f"Re-judging {len(cache_paths)} cached model(s): "
              f"{[p.stem for p in cache_paths]}")
        for cache_path in cache_paths:
            generation_cache[cache_path.stem] = json.loads(
                cache_path.read_text()
            )
    else:
        # Login to HuggingFace — required for gated WildJailbreak dataset
        # and any private model checkpoints
        # hf_token = os.environ.get("HF_TOKEN")
        # if hf_token:
        #     login(token=hf_token)
        # else:
        #     login()  # interactive prompt if no env var set

        # Load eval data once — shared across all models
        dataset = load_adversarial_harmful(args.max_samples)
        prompts = list(dataset["adversarial"])  # verify column name from printout
        print("Column names:", dataset.column_names)
        print("First example:", dataset[0])

        tactics_path = cache_dir / "tactics.json"  # define unconditionally first

        if "tactics" in dataset.column_names:
            tactics = list(dataset["tactics"])
            if not tactics_path.exists():
                tactics_path.write_text(json.dumps(tactics, indent=2))
                print(f"Tactics saved to {tactics_path}")
        else:
            tactics = None
            print("Warning: tactics column not present in eval split — "
              "tactic-level breakdown will be skipped")

        # ── Phase 1: Generate responses for all models ────────────────────────
        for config in MODEL_CONFIGS:
            cache_path = cache_dir / f"{config['name']}.json"

            # Resume from cache if generation already completed
            if cache_path.exists():
                cached_data = json.loads(cache_path.read_text())
                # Verify cache is non-empty and complete
                if (len(cached_data.get("responses", [])) == len(prompts)):
                    print(f"\n=== {config['name']}: cache found, "
                          f"skipping generation ===")
                    generation_cache[config["name"]] = cached_data
                    continue
                else:
                    print(f"\n=== {config['name']}: cache incomplete, "
                          f"regenerating ===")

            print(f"\n=== Generating: {config['name']} ({config['family']}) ===")

            # load_tokenizer handles pad_token and padding_side
            tokenizer = load_tokenizer(config["path"])

            model = AutoModelForCausalLM.from_pretrained(
                config["path"],
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            model.eval()

            # Sanity check: print first formatted prompt
            is_ra_sft = config.get("is_ra_sft", False)
            print("  Sample prompt format:")
            print("  " + format_prompt(
                prompts[0], config["family"], tokenizer, is_ra_sft=is_ra_sft
            )[:200])

            responses = generate_responses(
                model, tokenizer, prompts,
                args.batch_size, args.max_new_tokens,
                config["family"], is_ra_sft=is_ra_sft
            )
            generation_cache[config["name"]] = {
                "model": config["path"],
                "family": config["family"],
                "prompts": prompts,
                "responses": responses
            }

            # Cache to disk immediately — crash protection
            cache_path.write_text(
                json.dumps(generation_cache[config["name"]], indent=2)
            )
            print(f"  Cached to {cache_path}")

            del model
            torch.cuda.empty_cache()
            print(f"  VRAM cleared")

    # ── Phase 2: Load judge once, score all models ────────────────────────────
    print(f"\n=== Loading judge: {args.judge_path} "
          f"(revision={args.judge_revision or 'default'}) ===")

    # load_tokenizer handles pad_token and padding_side for judge too
    judge_tokenizer = load_tokenizer(args.judge_path, args.judge_revision)

    judge_model = AutoModelForCausalLM.from_pretrained(
        args.judge_path,
        revision=args.judge_revision,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    judge_model.eval()

    # Sanity check: print first judge prompt
    first_name = next(iter(generation_cache))
    first_data = generation_cache[first_name]
    print("  Sample judge prompt format:")
    print("  " + format_judge_prompt(
        first_data["prompts"][0], first_data["responses"][0], judge_tokenizer
    )[:300])

    all_results = {}
    all_raw_labels = {}
    for name, data in generation_cache.items():
        print(f"\n=== Judging: {name} ===")
        prompts = data["prompts"]
        responses = data["responses"]

        unsafe_flags, raw_labels = llamaguard_classify(
            judge_model, judge_tokenizer,
            prompts, responses,
            args.batch_size, max_length=args.judge_max_length
        )
        all_raw_labels[name] = raw_labels
        asr = sum(unsafe_flags) / len(unsafe_flags)
        print(f"  ASR: {asr:.4f} ({asr*100:.2f}%)")

        all_results[name] = {
            "model": data.get("model", name),
            "family": data.get("family"),
            "judge": args.judge_path,
            "judge_revision": args.judge_revision,
            "eval_set": "wildjailbreak_adversarial_harmful",
            "n_examples": len(unsafe_flags),
            "asr": asr,
            "examples": [
                {"prompt": p, "response": r, "unsafe": u}
                for p, r, u in zip(prompts, responses, unsafe_flags)
            ]
        }


    del judge_model
    torch.cuda.empty_cache()


    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*45}")
    print(f"{'Model':<20} {'N':>6} {'ASR':>8} {'Unsafe':>8}")
    print(f"{'-'*45}")
    for name, result in all_results.items():
        n_unsafe = sum(e["unsafe"] for e in result["examples"])
        print(f"{name:<20} {result['n_examples']:>6} "
              f"{result['asr']:>8.4f} {n_unsafe:>8}")
    print(f"{'='*45}")
    Path(args.output_path).write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved to {args.output_path}")
    raw_labels_path = Path(args.output_path).stem + "_raw_labels.json"
    Path(raw_labels_path).write_text(json.dumps(all_raw_labels, indent=2))
    print(f"Raw labels saved to {raw_labels_path}")


if __name__ == "__main__":
    main()