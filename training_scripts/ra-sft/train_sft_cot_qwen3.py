import os
import re
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

# --- 1. CONFIGURATION ---
_HERE = os.path.dirname(os.path.abspath(__file__))
model_id = "Qwen/Qwen3-8B"
dataset_path = os.path.join(_HERE, "training_data.jsonl")
output_dir = os.path.join(_HERE, "qwen3-8b-safetysft_cot")

# --- 2. LOAD DATASET ---
dataset = load_dataset("json", data_files=dataset_path, split="train")

# --- 3. CoT FORMATTING FUNCTION ---
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    reasonings = examples["reasoning"]
    raw_outputs = examples["output"]
    texts = []

    for instruction, input_text, reasoning, raw_output in zip(instructions, inputs, reasonings, raw_outputs):
        # Clean the output: strip any existing <think> tags from the 'output' field
        # so we control the structure exactly as planned.
        clean_output = re.sub(r'<think>.*?</think>\s*', '', raw_output, flags=re.DOTALL).strip()

        # Qwen3 uses <|im_start|>/<|im_end|> instead of Llama-3's
        # <|begin_of_text|>/<|start_header_id|>/<|eot_id|>. No BOS token
        # is prepended by convention for Qwen3 (unlike Llama's
        # <|begin_of_text|>) -- the tokenizer's own encoding step handles
        # any required special-token insertion at train time.
        text = f"""<|im_start|>user
{instruction}
{input_text if input_text else ""}<|im_end|>
<|im_start|>assistant
<think>
{reasoning.strip()}
</think>
{clean_output}<|im_end|>"""
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

# --- 4. LOAD MODEL & TOKENIZER ---
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    use_cache=False,
)
model.gradient_checkpointing_enable()

# --- 5. TRAINING ARGUMENTS ---
# No DeepSpeed: single-GPU training, so ZeRO sharding/offload isn't needed.
# Gradient checkpointing above + bf16 below handle memory on one A100/L40S.
# trl >= 1.0.0's SFTConfig no longer accepts warmup_ratio, only warmup_steps
approx_total_steps = (len(dataset) * 3) // 128
warmup_steps = max(1, int(0.03 * approx_total_steps))

training_args = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=1,      # Optimized for A100 80GB
    gradient_accumulation_steps=128,      # Effective batch size of 128
    learning_rate=1e-5,
    num_train_epochs=3,                 # Full 3-epoch run for CoT
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    save_only_model=True,               # Fix for the 'save_checkpoint' error
    bf16=True,
    tf32=True,
    lr_scheduler_type="cosine",
    warmup_steps=warmup_steps,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_pin_memory=False,
    dataset_text_field="text",
    max_length=2048,
    packing=False,
    seed=42,
    data_seed=42
)

# --- 6. SFT TRAINER ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
)

# --- 7. START TRAINING ---
print("Starting fresh CoT fine-tuning on Qwen3-8B (single-GPU, no DeepSpeed)...")
trainer.train()

# --- 8. FINAL SAVE ---
print("Saving final model...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Success! CoT model saved to {output_dir}")