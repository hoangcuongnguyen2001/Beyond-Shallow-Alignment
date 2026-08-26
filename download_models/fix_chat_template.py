# fix_tokenizer.py — run once per checkpoint
from transformers import AutoTokenizer

# Llama-3.1 instruct chat template
# This is the canonical template from Meta's instruct model
LLAMA31_CHAT_TEMPLATE = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.first and messages[0]['role'] != 'system' %}"
    "{% set content = bos_token + content %}"
    "{% endif %}"
    "{{ content }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{% endif %}"
)

GEMMA2_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% if messages[0]['role'] == 'system' %}"
    "{{ raise_exception('System role not supported') }}"
    "{% endif %}"
    "{% for message in messages %}"
    "{% if (message['role'] == 'user') != (loop.first) %}"
    "{{ raise_exception('Conversation roles must alternate "
    "user/assistant/user/assistant/...') }}"
    "{% endif %}"
    "{% if (message['role'] == 'assistant') %}"
    "{% set role = 'model' %}"
    "{% else %}"
    "{% set role = message['role'] %}"
    "{% endif %}"
    "{{ '<start_of_turn>' + role + '\\n' + "
    "message['content'] | trim + '<end_of_turn>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{'<start_of_turn>model\\n'}}"
    "{% endif %}"
)

checkpoints = [
    "models/gemma-sft",
    "models/gemma-ra-sft", 
    "models/gemma-orpo",
]

for path in checkpoints:
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.chat_template = GEMMA2_CHAT_TEMPLATE
    tokenizer.save_pretrained(path)
    print(f"Fixed: {path}")

# Verify
tokenizer = AutoTokenizer.from_pretrained("models/gemma-sft")
test = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Hello"}],
    tokenize=False,
    add_generation_prompt=True
)
print(repr(test))
# Should produce:
# '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n
#  Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'

try:
    tokenizer.apply_chat_template(
        [{"role": "system", "content": "Be helpful"},
         {"role": "user",   "content": "Hello"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    print("ERROR: system role should have raised exception")
except Exception as e:
    print(f"System role correctly rejected: {e}")