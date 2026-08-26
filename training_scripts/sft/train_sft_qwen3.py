import os
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

# ---------------- Configuration ----------------
_HERE = os.path.dirname(os.path.abspath(__file__))
model_id = "Qwen/Qwen3-8B-Base"
dataset_path = os.path.join(_HERE, "safety_sft_dataset.jsonl")
output_dir = os.path.join(_HERE, "qwen3-8b-safetysft")

# ---------------- Tokenizer ----------------
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ---------------- Model ----------------
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="sdpa",
)
model.config.use_cache = False
model.gradient_checkpointing_enable()

# ---------------- Dataset ----------------
with open(dataset_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]
dataset = Dataset.from_list(data)
print(f"Dataset size: {len(dataset)}")

def formatting_prompts_func(example):
    user = example["instruction"]
    if example["input"]:
        user += "\n" + example["input"]
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user},
        {"role": "assistant", "content": example["output"]},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

# trl >= 1.0.0's SFTConfig no longer accepts warmup_ratio, only warmup_steps
approx_total_steps = (len(dataset) * 3) // 128
warmup_steps = max(1, int(0.1 * approx_total_steps))

training_args = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=128,
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
    report_to="none",
    push_to_hub=False,
    dataloader_pin_memory=False,
    max_length=2048,
    packing=False,
    seed=42,
    data_seed=42
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    args=training_args,
)

trainer.train()
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")
