import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Detect device (works on CUDA, Apple Silicon MPS, or CPU)
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

model_name = "unsloth/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Example multi-turn chat
messages = [
    {"role": "system",    "content": "You are a helpful tutor."},
    {"role": "user",      "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language."},
    {"role": "user",      "content": "Who created it?"},
]

# return_dict=True returns input_ids + attention_mask (transformers >= 4.44)
tokens = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    return_dict=True
).to(device)

output_tokens = model.generate(**tokens, max_new_tokens=100)
result = tokenizer.decode(output_tokens[0])
print(result)