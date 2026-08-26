import os
import json
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.experimental.orpo import ORPOTrainer, ORPOConfig

# ── Configuration ──────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
model_id = os.path.join(_HERE, "qwen3-8b-weights")
dataset_path = os.path.join(_HERE, "safety_orpo_dataset.jsonl")
output_dir = os.path.join(_HERE, "qwen3-8b-safetyorpo")

_SYSTEM = "You are a helpful assistant."


# ── Load tokenizer ─────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
tokenizer.padding_side = "right"


def _apply_chat_template(messages, add_generation_prompt: bool) -> str:
    # enable_thinking=False so this run sits on the same footing as the
    # Llama/Gemma ORPO runs (no <think> scaffold populated in targets).
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )


def _selftest_template() -> None:
    """
    Verifies that full_conversation_render.startswith(prompt_only_render)
    holds for Qwen3 with enable_thinking=False. This assumption is what
    the Gemma script relies on to split 'chosen'/'rejected' out of the
    full rendered string. Qwen3 inserts an empty <think>\\n\\n</think>\\n\\n
    stub at the assistant-turn boundary; whether it renders identically
    in both the generation-prompt and full-conversation paths is a
    template-implementation detail, not a guarantee. Fail loudly here
    rather than silently mis-splitting the real dataset.
    """
    probe_user = "test prompt"
    probe_assistant = "test response"

    prompt_only = _apply_chat_template(
        [{"role": "user", "content": probe_user}],
        add_generation_prompt=True,
    )
    full = _apply_chat_template(
        [
            {"role": "user", "content": probe_user},
            {"role": "assistant", "content": probe_assistant},
        ],
        add_generation_prompt=False,
    )

    print("\n--- Qwen3 template self-test ---")
    print("prompt_only:", repr(prompt_only))
    print("full:      ", repr(full))
    print("startswith match:", full.startswith(prompt_only))
    print("---------------------------------\n")

    if not full.startswith(prompt_only):
        raise ValueError(
            "Qwen3 chat template self-test FAILED: the full-conversation "
            "render does not start with the prompt-only render under "
            "enable_thinking=False. This means the prefix-splitting logic "
            "below (same mechanism used for the Gemma ORPO script) will "
            "mis-split prompt/chosen/rejected for every row. Inspect the "
            "two printed strings above, find where they diverge (likely "
            "around the <think> stub), and adjust apply_chat_template() "
            "to strip/align that stub explicitly before rerunning on the "
            "full dataset. Do not proceed to dataset.map() until this "
            "passes."
        )


_selftest_template()


# ── Load model in bfloat16 ─────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="sdpa",
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
for i in range(min(3, len(df))):
    print(f"\nPrompt: {df['prompt'].iloc[i][:100]}...")
    print(f"Chosen: {df['chosen'].iloc[i][:150]}...")
    print(f"Rejected: {df['rejected'].iloc[i][:150]}...")
    print("-" * 60)

dataset = Dataset.from_pandas(df)


# ── Apply Qwen3 chat template (prefix splitting) ───────────
# ORPOTrainer expects:
#   prompt: prefix up to (and including) the assistant turn start
#   chosen: preferred completion only (assistant content)
#   rejected: rejected completion only (assistant content)
def apply_chat_template(example):
    user_content = f"{_SYSTEM}\n\n{example['prompt']}".strip()

    # Prompt prefix ends at the assistant turn start.
    prompt_text = _apply_chat_template(
        [{"role": "user", "content": user_content}],
        add_generation_prompt=True,
    )

    # Full strings: user + assistant completion.
    full_chosen = _apply_chat_template(
        [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example["chosen"]},
        ],
        add_generation_prompt=False,
    )
    full_rejected = _apply_chat_template(
        [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example["rejected"]},
        ],
        add_generation_prompt=False,
    )

    if not full_chosen.startswith(prompt_text):
        raise ValueError(
            "Qwen3 chat template prefix mismatch for `chosen`. The "
            "self-test at startup should have caught this class of "
            "issue — if it passed but this row fails, the divergence "
            "may be content-dependent (e.g. specific characters "
            "interacting with the template). Inspect this row directly."
        )
    if not full_rejected.startswith(prompt_text):
        raise ValueError(
            "Qwen3 chat template prefix mismatch for `rejected`. See "
            "note above for `chosen`."
        )

    chosen_text = full_chosen[len(prompt_text):].strip()
    rejected_text = full_rejected[len(prompt_text):].strip()
    return {"prompt": prompt_text, "chosen": chosen_text, "rejected": rejected_text}


dataset = dataset.map(apply_chat_template)
print(f"\nDataset after template application: {len(dataset)}")


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
    beta=0.1,  # odds ratio penalty weight
    max_length=2048,  # maximum total sequence length
    max_completion_length=1536,  # maximum completion length
    seed=42,
    data_seed=42
)


trainer = ORPOTrainer(
    model=model,
    args=orpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Qwen3 ORPO model saved to {output_dir}")
