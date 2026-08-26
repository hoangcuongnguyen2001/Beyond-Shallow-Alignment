import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

MODEL_PATH = "models/llama-ra-sft"

device = "cuda"

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model.eval()

# -------------------------------------------------
# LOAD ONE MMLU SAMPLE
# -------------------------------------------------

ds = load_dataset("cais/mmlu", "professional_law", split="test")

sample = ds[0]

choices = "\n".join(
    f"{chr(65+i)}. {c}"
    for i, c in enumerate(sample["choices"])
)

prompt = f"""
{sample['question']}

{choices}

Answer with only A, B, C, or D.
"""

answer = chr(65 + sample["answer"])

# -------------------------------------------------
# CHAT TEMPLATE
# -------------------------------------------------

formatted = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(
    formatted,
    return_tensors="pt",
).to(device)

# -------------------------------------------------
# GENERATE
# -------------------------------------------------

with torch.no_grad():

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

decoded = tokenizer.decode(
    new_tokens,
    skip_special_tokens=True,
).strip()

# -------------------------------------------------
# EXTRACT ANSWER
# -------------------------------------------------

def extract_answer(text):

    text = text.upper()

    patterns = [
        r"\b([ABCD])\b",
        r"ANSWER\s*[:\-]?\s*([ABCD])",
    ]

    for p in patterns:
        m = re.search(p, text)

        if m:
            return m.group(1)

    return ""

predicted = extract_answer(decoded)

# -------------------------------------------------
# PRINT DEBUG
# -------------------------------------------------

print("\n===================")
print("PROMPT")
print("===================")
print(prompt)

print("\n===================")
print("RAW OUTPUT")
print("===================")
print(decoded)

print("\n===================")
print("PREDICTED")
print("===================")
print(predicted)

print("\n===================")
print("GROUND TRUTH")
print("===================")
print(answer)