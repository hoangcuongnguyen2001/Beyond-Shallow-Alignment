
import os
import re
import json
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.experimental.orpo import ORPOTrainer, ORPOConfig
 
# ── Configuration ──────────────────────────────────────────
# Paths are relative to the repository root (D:/huggingface/)
_HERE = os.path.dirname(os.path.abspath(__file__))
model_id = os.path.join(_HERE, "gemma-2-9b-weights")
# Load tokenizer from IT checkpoint (chat template) but model from base weights
it_tokenizer_id = os.path.join(_HERE, "gemma-2-9b-it-tokenizer")
dataset_path = os.path.join(_HERE, "safety_orpo_dataset.jsonl")
output_dir = os.path.join(_HERE, "gemma-2-9b-safetyorpo")
 
_SYSTEM = "You are a helpful assistant."
 
# ── Load tokenizer from IT checkpoint ─────────────────────
# Base model has no chat_template; IT tokenizer shares the same
# vocabulary and tokenizer but adds the Gemma 2 chat template.
# Model weights still come from the base checkpoint below.
tokenizer = AutoTokenizer.from_pretrained(it_tokenizer_id)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
tokenizer.padding_side = "right"
print(f"✓ Tokenizer loaded from IT checkpoint: {it_tokenizer_id}")
print(f"✓ Chat template present: {tokenizer.chat_template is not None}")
 
# ── Load model in bfloat16 ─────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)
model.config.use_cache = False
model.gradient_checkpointing_enable()
print(f"✓ Model loaded from base checkpoint: {model_id}")
 
# ── Load dataset ───────────────────────────────────────────
data = []
with open(dataset_path, "r") as f:
    for line in f:
        data.append(json.loads(line.strip()))
 
df = pd.DataFrame(data)
print(f"Total ORPO pairs: {len(df)}")
print(f"Unique prompts: {df['prompt'].nunique()}")
 
# Spot check before training
print("\nSample pair verification:")
for i in range(min(3, len(df))):
    print(f"\nPrompt: {df['prompt'].iloc[i][:100]}...")
    print(f"Chosen: {df['chosen'].iloc[i][:150]}...")
    print(f"Rejected: {df['rejected'].iloc[i][:150]}...")
    print("-" * 60)
 
dataset = Dataset.from_pandas(df)
 
# ── Apply Gemma chat template (prefix splitting) ──────────
# ORPOTrainer expects:
#   prompt: prefix up to (and including) the assistant turn start
#   chosen: preferred completion only (assistant content)
#   rejected: rejected completion only (assistant content)
def apply_chat_template(example):
    user_content = f"{_SYSTEM}\n\n{example['prompt']}".strip()
 
    # Prompt prefix ends at the assistant turn start
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )
 
    # Full strings: user + assistant completion
    full_chosen = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example["chosen"]},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    full_rejected = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example["rejected"]},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
 
    if not full_chosen.startswith(prompt_text):
        raise ValueError(
            "Gemma chat template prefix mismatch for `chosen`. "
            "The produced `prompt_text` is not a strict prefix of `full_chosen`."
        )
    if not full_rejected.startswith(prompt_text):
        raise ValueError(
            "Gemma chat template prefix mismatch for `rejected`. "
            "The produced `prompt_text` is not a strict prefix of `full_rejected`."
        )
 
    chosen_text = full_chosen[len(prompt_text):]
    rejected_text = full_rejected[len(prompt_text):]
 
    return {
        "prompt": prompt_text,
        "chosen": chosen_text.strip(),
        "rejected": rejected_text.strip()
    }
 
dataset = dataset.map(apply_chat_template)
print(f"\nDataset after template application: {len(dataset)}")
print("Columns:", dataset.column_names)
 
# ── ORPO configuration ─────────────────────────────────────
# trl >= 1.0.0's config classes no longer accept warmup_ratio, only warmup_steps
approx_total_steps = (len(dataset) * 3) // 128
warmup_steps = max(1, int(0.1 * approx_total_steps))

orpo_config = ORPOConfig(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=128,  # effective batch = 128
    learning_rate=1e-5,
    num_train_epochs=3,
    warmup_steps=warmup_steps,
    weight_decay=0.01,
    logging_steps=5,
    bf16=True,
    tf32=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    save_strategy="epoch",
    save_total_limit=1,
    push_to_hub=False,
    report_to="none",
    dataloader_pin_memory=False,
    # ORPO-specific parameters
    beta=0.1,              # odds ratio penalty weight
    max_length=2048,       # maximum total sequence length
    max_completion_length=1536,
    seed=42,
    data_seed=42
)
 
# ── Train ──────────────────────────────────────────────────
trainer = ORPOTrainer(
    model=model,
    args=orpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)
 


# # ── Diagnostic run (50 steps) ──────────────────────────────────────────────────
# # Run this first to confirm no OOM and loss is decreasing before full training.
# # Comment out training_args.max_steps and change logging_steps back for full run.
# orpo_config.max_steps = 50
# orpo_config.logging_steps = 1


# print("=== Starting diagnostic run (50 steps) ===")
# print("Check for:")
# print("1. Loss between 1.0-3.0 and decreasing")
# print("2. No OOM errors")
# print("3. CUDA utilisation > 0")
trainer.train()
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Gemma ORPO model saved to {output_dir}")

# trainer.train()

# print("=== Diagnostic complete ===")
# print("If loss is healthy, comment out max_steps and resubmit for full training")
 