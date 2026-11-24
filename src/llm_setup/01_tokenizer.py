import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Detect device (works on CUDA, Apple Silicon MPS, or CPU)
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# We are going to use Llama 3.2
model_name = "unsloth/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

sample_text = "Hello world"
tokens = tokenizer(sample_text)
print(tokens)  # By default tokens are returned as a list, but the model expects tensors

tokens = tokenizer(sample_text, return_tensors="pt").to(device)
print(tokens)  # Model accepts this format