import os
import torch
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from trl import SFTTrainer, SFTConfig


_HERE           = os.path.dirname(os.path.abspath(__file__))
model_id        = os.path.join(_HERE, "gemma-2-9b-weights")
it_tokenizer_id = os.path.join(_HERE, "gemma-2-9b-it-tokenizer")
dataset_path    = os.path.join(_HERE, "safety_sft_dataset.jsonl")
output_dir      = os.path.join(_HERE, "gemma-2-9b-safetysft")
persistent_dir  = None   # no persistent copy on Vast.ai — download manually

_SYSTEM = "You are a helpful assistant."

# ── Load tokenizer ─────────────────────────────────────────────────────────────
# Load base model tokenizer, then borrow chat_template from IT tokenizer.
# Safe because both share identical vocabulary — the chat_template only
# affects message formatting, not token embeddings or model weights.
tokenizer = AutoTokenizer.from_pretrained("/workspace/gemma-2-9b-it-tokenizer")
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
tokenizer.padding_side = "right"
print("✓ Tokenizer loaded with chat template")

# ── Load model in bfloat16 ─────────────────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    attn_implementation="sdpa",
)
model.config.use_cache = False
model.gradient_checkpointing_enable()

# ── Load and validate dataset ──────────────────────────────────────────────────
data = []
skipped = 0
with open(dataset_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        clean_line = line.strip()
        if not clean_line:
            continue
        try:
            data.append(json.loads(clean_line))
        except json.JSONDecodeError:
            print(f"Skipping malformed line {i+1}")
            skipped += 1

dataset = Dataset.from_list(data)
print(f"Dataset size: {len(dataset)} (skipped {skipped} malformed lines)")

# Validate expected columns exist
expected_cols = {"instruction", "input", "output"}
actual_cols = set(dataset.column_names)
if not expected_cols.issubset(actual_cols):
    raise ValueError(
        f"Dataset missing expected columns. "
        f"Expected: {expected_cols}, Got: {actual_cols}"
    )
print(f"Dataset columns confirmed: {dataset.column_names}")
print(f"Sample row: {dataset[0]}")

# ── Chat template formatter ────────────────────────────────────────────────────
def formatting_prompts_func(example):
    instruction = example["instruction"]
    inp = example["input"] if example["input"] else ""
    output = example["output"]

    user_body = f"{instruction}\n{inp}".strip()
    user_content = f"{_SYSTEM}\n\n{user_body}"

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return text

# ── Training configuration ─────────────────────────────────────────────────────
# trl >= 1.0.0's SFTConfig no longer accepts warmup_ratio, only warmup_steps
approx_total_steps = (len(dataset) * 3) // 128
warmup_steps = max(1, int(0.1 * approx_total_steps))
training_args = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=128,   # effective batch = 128, matches Llama runs
    learning_rate=1e-5,                # matches Llama SFT
    num_train_epochs=3,                # matches Llama runs
    warmup_steps=warmup_steps,
    weight_decay=0.01,
    logging_steps=5,
    bf16=True,
    tf32=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    # Save every 100 steps — critical for Gadi interruptible jobs
    # and for having intermediate checkpoints (lesson from DPO overfitting)
    save_strategy="epoch",
    save_total_limit=1,                
    push_to_hub=False,
    report_to="none",
    dataloader_pin_memory=False,
    max_length=2048,
    packing=False,
    seed=42,
    data_seed=42
)

# ── Diagnostic run (50 steps) ──────────────────────────────────────────────────
# Run this first to confirm no OOM and loss is decreasing before full training.
# Comment out training_args.max_steps and change logging_steps back for full run.
# training_args.max_steps = 50
# training_args.logging_steps = 1

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    formatting_func=formatting_prompts_func,
    args=training_args,
)


trainer.train()  # resumes if job was interrupted
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")

# # Copy to persistent gdata storage (Gadi only)
# # On Vast.ai: download the checkpoint manually instead
# if USE_GADI and persistent_dir:
#     import shutil
#     os.makedirs(persistent_dir, exist_ok=True)
#     shutil.copytree(output_dir, persistent_dir, dirs_exist_ok=True)
#     print(f"Copied to persistent storage: {persistent_dir}")
# elif not USE_GADI:
#     print(f"Vast.ai: download checkpoint from {output_dir} manually")
#     print(f"tar -czvf gemma-2-9b-safetysft.tar.gz {output_dir}")
