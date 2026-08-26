import os
import re
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

# --- 1. CONFIGURATION ---
# Paths are relative to the repository root (D:/huggingface/)
# Run this script from that directory: python train_sft_cot.py
_HERE = os.path.dirname(os.path.abspath(__file__))
model_id = os.path.join(_HERE, "llama-3.1-8b-weights")
dataset_path = os.path.join(_HERE, "training_data.jsonl")
output_dir = os.path.join(_HERE, "llama-3-8.1b-safetysft_cot")

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
        # Clean the output: Strip any existing <think> tags from the 'output' field
        # to ensure we control the structure exactly as planned.
        clean_output = re.sub(r'<think>.*?</think>\s*', '', raw_output, flags=re.DOTALL).strip()
        
        # Build the Llama-3 Template with specific CoT tagging
        text = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}
{input_text if input_text else ""}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

<think>
{reasoning.strip()}
</think>
{clean_output}<|eot_id|>"""
        texts.append(text)
    return { "text": texts }

dataset = dataset.map(formatting_prompts_func, batched=True)

# --- 4. LOAD MODEL & TOKENIZER ---
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_cache=False 
)

# --- 5. TRAINING ARGUMENTS ---
# trl >= 1.0.0's SFTConfig no longer accepts warmup_ratio, only warmup_steps
approx_total_steps = (len(dataset) * 3) // 128
warmup_steps = max(1, int(0.03 * approx_total_steps))

training_args = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=1,      # Optimized for A100 80GB
    gradient_accumulation_steps=128,      # Effective batch size of 32
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
    deepspeed=os.path.join(_HERE, "ds_config.json"),
    # SFT-specific args (moved here from SFTTrainer in TRL >= 1.0.0)
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
print("🚀 Starting FRESH CoT Fine-Tuning on Llama-3-8B...")
trainer.train()

# --- 8. FINAL SAVE ---
print("Saving final model...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✅ Success! CoT Model saved to {output_dir}")