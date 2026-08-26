import os
import re
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from trl import SFTTrainer, SFTConfig

# ── Configuration ──────────────────────────────────────────
# Place Gemma 2 weights under `gemma-2-9b-weights/` (e.g. snapshot of
# `google/gemma-2-9b-it`). The tokenizer must include a `chat_template`.
_HERE = os.path.dirname(os.path.abspath(__file__))
model_id = os.path.join(_HERE, "gemma-2-9b-weights")
it_tokenizer_id = os.path.join(_HERE, "gemma-2-9b-it-tokenizer")
dataset_path = os.path.join(_HERE, "training_data.jsonl")
output_dir = os.path.join(_HERE, "gemma-2-9b-safetysft-cot")

# ── Load tokenizer ─────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(it_tokenizer_id)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
tokenizer.padding_side = "right"

# ── Load model in bfloat16 ─────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},          # explicit single GPU
    attn_implementation="sdpa",
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

# ── Chat template formatter (Gemma 2 + CoT) ───────────────
# Gemma 2 uses alternating user/model turns with `<start_of_turn>` /
# `<end_of_turn>` markers. `apply_chat_template` keeps this in sync
# with the tokenizer's `chat_template`.
#
# The system prompt is folded into the user turn (Gemma 2 has no
# dedicated system role). The reasoning field is wrapped in
# `<think>...</think>` before the final answer.
_SYSTEM = "You are a helpful assistant."


def formatting_prompts_func(example):
    instruction = example["instruction"]
    input_text = example["input"] if example["input"] else ""
    reasoning = example["reasoning"]
    raw_output = example["output"]

    # Strip any pre-existing <think> blocks so we control the exact structure.
    clean_output = re.sub(
        r"<think>.*?</think>\s*", "", raw_output, flags=re.DOTALL
    ).strip()

    user_body = f"{instruction}\n{input_text}".strip()
    user_content = f"{_SYSTEM}\n\n{user_body}"

    # Gemma chat template wraps the assistant turn; we only inject CoT tags.
    assistant_content = f"<think>\n{reasoning.strip()}\n</think>\n{clean_output}"

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError(
            "Tokenizer has no chat_template. Use a Gemma 2 instruction-tuned "
            "tokenizer (e.g. from google/gemma-2-9b-it) or set tokenizer.chat_template."
        )

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


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
    save_total_limit=1,
    push_to_hub=False,
    report_to="none",
    dataloader_pin_memory=False,
    # Gemma 2 supports up to 8k–16k tokens depending on checkpoint; raise if VRAM allows.
    max_length=2048,
    packing=False,
    seed=42,
    data_seed=42
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    formatting_func=formatting_prompts_func,
    args=training_args,
)

print("Starting Gemma-2 CoT Fine-Tuning...")
trainer.train()

print("Saving final model...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")
