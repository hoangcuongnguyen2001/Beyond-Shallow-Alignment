import torch
import json
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from trl.experimental.orpo import ORPOTrainer, ORPOConfig

# ── Configuration ──────────────────────────────────────────
model_id = "/workspace/llama-weights"
dataset_path = "/workspace/safety_orpo_dataset.jsonl"
output_dir = "./safety_orpo_output"

# ── Load tokenizer ─────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ── Load model in bfloat16 ─────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)
model.config.use_cache = False
model.gradient_checkpointing_enable()

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
for i in range(3):
    print(f"\nPrompt: {df['prompt'].iloc[i][:100]}...")
    print(f"Chosen: {df['chosen'].iloc[i][:150]}...")
    print(f"Rejected: {df['rejected'].iloc[i][:150]}...")
    print("-"*60)

dataset = Dataset.from_pandas(df)

# ── Apply chat template ────────────────────────────────────
# ORPOTrainer expects prompt/chosen/rejected fields
# with chat template applied to all three
def apply_chat_template(example):
    # Format prompt as user message
    prompt_text = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"You are a helpful assistant."
        f"<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{example['prompt']}"
        f"<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    # Chosen and rejected are assistant responses only
    chosen_text = f"{example['chosen']}<|eot_id|>"
    rejected_text = f"{example['rejected']}<|eot_id|>"
    
    return {
        'prompt': prompt_text,
        'chosen': chosen_text,
        'rejected': rejected_text
    }

dataset = dataset.map(apply_chat_template)
print(f"\nDataset after template application: {len(dataset)}")

# ── ORPO Configuration ─────────────────────────────────────
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
    max_completion_length=1536, # maximum prompt length
    seed=42,
    data_seed=42
)

# # ── Diagnostic run first ───────────────────────────────────
# print("\nRunning 50-step diagnostic...")
# orpo_config.max_steps = 50
# orpo_config.logging_steps = 1

trainer = ORPOTrainer(
    model=model,
    args=orpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# trainer.train()
# print("\nDiagnostic complete.")
# print("Check for:")
# print("1. loss between 1.0-3.0 and decreasing")
# print("2. rewards/chosen and rewards/rejected logged")
# print("3. log_odds_ratio logged and non-zero")
# print("4. No OOM errors")

# import os
# import json
# import torch
# import pandas as pd
# from datasets import Dataset
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from trl.experimental.orpo import ORPOTrainer, ORPOConfig

# # ── Configuration ──────────────────────────────────────────
# # Paths are relative to the repository root (D:/huggingface/)
# # Run this script from that directory: python train_orpo.py
# _HERE = os.path.dirname(os.path.abspath(__file__))
# model_id = os.path.join(_HERE, "gemma-2-9b-weights")
# dataset_path = os.path.join(_HERE, "safety_orpo_dataset.jsonl")
# output_dir = os.path.join(_HERE, "gemma-2-9b-safetyorpo")

# _SYSTEM = "You are a helpful assistant."


# # ── Load tokenizer ─────────────────────────────────────────
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
# tokenizer.padding_side = "right"


# def _apply_chat_template(messages, add_generation_prompt: bool) -> str:
#     # Some Gemma tokenizers use role name `model` instead of `assistant`.
#     # Try `assistant` first; fall back to `model` if needed.
#     try:
#         return tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=add_generation_prompt,
#         )
#     except Exception:
#         fixed = [
#             {"role": ("model" if m.get("role") == "assistant" else m.get("role")), "content": m.get("content")}
#             for m in messages
#         ]
#         return tokenizer.apply_chat_template(
#             fixed,
#             tokenize=False,
#             add_generation_prompt=add_generation_prompt,
#         )


# # ── Load model in bfloat16 ─────────────────────────────────
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
#     attn_implementation="sdpa",
# )
# model.config.use_cache = False
# model.gradient_checkpointing_enable()


# # ── Load dataset ───────────────────────────────────────────
# data = []
# with open(dataset_path, "r") as f:
#     for line in f:
#         data.append(json.loads(line.strip()))

# df = pd.DataFrame(data)
# print(f"Total ORPO pairs: {len(df)}")
# print(f"Unique prompts: {df['prompt'].nunique()}")

# # Spot check before training
# print("\nSample pair verification:")
# for i in range(min(3, len(df))):
#     print(f"\nPrompt: {df['prompt'].iloc[i][:100]}...")
#     print(f"Chosen: {df['chosen'].iloc[i][:150]}...")
#     print(f"Rejected: {df['rejected'].iloc[i][:150]}...")
#     print("-" * 60)

# dataset = Dataset.from_pandas(df)


# # ── Apply Gemma chat template (prefix splitting) ──────────
# # ORPOTrainer expects:
# #   prompt: prefix up to (and including) the assistant turn start
# #   chosen: preferred completion only (assistant content)
# #   rejected: rejected completion only (assistant content)
# def apply_chat_template(example):
#     user_content = f"{_SYSTEM}\n\n{example['prompt']}".strip()

#     # Prompt prefix ends at the assistant turn start.
#     prompt_text = _apply_chat_template(
#         [{"role": "user", "content": user_content}],
#         add_generation_prompt=True,
#     )

#     # Full strings: user + assistant completion.
#     full_chosen = _apply_chat_template(
#         [
#             {"role": "user", "content": user_content},
#             {"role": "assistant", "content": example["chosen"]},
#         ],
#         add_generation_prompt=False,
#     )
#     full_rejected = _apply_chat_template(
#         [
#             {"role": "user", "content": user_content},
#             {"role": "assistant", "content": example["rejected"]},
#         ],
#         add_generation_prompt=False,
#     )

#     if not full_chosen.startswith(prompt_text):
#         raise ValueError(
#             "Gemma chat template prefix mismatch for `chosen`."
#         )
#     if not full_rejected.startswith(prompt_text):
#         raise ValueError(
#             "Gemma chat template prefix mismatch for `rejected`."
#         )

#     chosen_text = full_chosen[len(prompt_text) :].strip()
#     rejected_text = full_rejected[len(prompt_text) :].strip()
#     return {"prompt": prompt_text, "chosen": chosen_text, "rejected": rejected_text}


# dataset = dataset.map(apply_chat_template)
# print(f"\nDataset after template application: {len(dataset)}")


# # ── ORPO configuration ─────────────────────────────────────
# approx_total_steps = (len(dataset) * 3) // 128
# warmup_steps = max(1, int(0.1 * approx_total_steps))
# orpo_config = ORPOConfig(
#     output_dir=output_dir,
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=128,  # effective batch = 128
#     learning_rate=1e-5,
#     num_train_epochs=3,
#     warmup_steps=warmup_steps,
#     weight_decay=0.01,
#     logging_steps=5,
#     bf16=True,
#     tf32=True,
#     gradient_checkpointing=True,
#     gradient_checkpointing_kwargs={"use_reentrant": False},
#     save_strategy="epoch",
#     save_total_limit=1,
#     push_to_hub=False,
#     report_to="none",
#     dataloader_pin_memory=False,
#     beta=0.1,  # odds ratio penalty weight
#     max_length=2048,  # maximum total sequence length
#     max_completion_length=1536,  # maximum prompt length
# )


# trainer = ORPOTrainer(
#     model=model,
#     args=orpo_config,
#     train_dataset=dataset,
#     processing_class=tokenizer,
# )

# trainer.train()
# trainer.save_model(output_dir)
# tokenizer.save_pretrained(output_dir)
# print(f"Gemma ORPO model saved to {output_dir}")