import os
import torch
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from trl import SFTTrainer, SFTConfig

# ── Configuration ──────────────────────────────────────────
# Paths are relative to the repository root (D:/huggingface/)
# Run this script from that directory: python train_sft.py
_HERE = os.path.dirname(os.path.abspath(__file__))
model_id = os.path.join(_HERE, "llama-3.1-8b-weights")
dataset_path = os.path.join(_HERE, "safety_sft_dataset.jsonl")
output_dir = os.path.join(_HERE, "llama-3.1-8b-safetysft")

# ── Load tokenizer ─────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ── Load model in bfloat16 ─────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,   # critical: bfloat16 not float16
    device_map="auto",
    attn_implementation="sdpa"
)
model.config.use_cache = False
model.gradient_checkpointing_enable()

# ── Load and format dataset ────────────────────────────────
data = []
with open(dataset_path, "r") as f:
    for line in f:
        data.append(json.loads(line.strip()))

dataset = Dataset.from_list(data)
print(f"Dataset size: {len(dataset)}")

# ── Chat template formatter ────────────────────────────────
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        instruction = example['instruction'][i]
        inp = example['input'][i] if example['input'][i] else ""
        output = example['output'][i]
        
        # Combine instruction and input if input exists
        user_content = f"{instruction}\n{inp}".strip()
        
        text = (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n"
            f"You are a helpful assistant."
            f"<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_content}"
            f"<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{output}"
            f"<|eot_id|>"
        )
        output_texts.append(text)
    return output_texts

# ── Training arguments ─────────────────────────────────────
# trl >= 1.0.0's SFTConfig no longer accepts warmup_ratio, only warmup_steps
approx_total_steps = (len(dataset) * 3) // 128
warmup_steps = max(1, int(0.1 * approx_total_steps))
training_args = SFTConfig(
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
    save_total_limit=1,           # saves disk space
    push_to_hub=False,
    report_to="none",
    dataloader_pin_memory=False,  # prevents memory issues
    # SFT-specific args (moved here from SFTTrainer in TRL >= 1.0.0)
    max_length=2048,
    packing=False,
    seed=42,
    data_seed=42
)

# Reset for full training after diagnostic confirms correct setup
trainer_full = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    args=training_args,
)

trainer_full.train()
trainer_full.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")