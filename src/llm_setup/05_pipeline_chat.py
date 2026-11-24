import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Detect device (works on CUDA, Apple Silicon MPS, or CPU)
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# --- Part 1: pipeline can also download and load a model by itself ---
model_pipeline2 = pipeline("text-generation", model="gpt2")
result = model_pipeline2("Once upon a time")
print(result)

# --- Part 2: Pipelines also handle the chat template automatically ---
model_name = "unsloth/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

model_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=500,
    temperature=0.1,
    top_k=10,
    top_p=0.85,
)

messages = [
    {"role": "system", "content": "You are a helpful medical assistant chatbot."},
    {"role": "user",   "content": "How are you?"},
]

response = model_pipeline(messages)
print(response)