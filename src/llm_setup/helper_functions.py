import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Detect device (works on CUDA, Apple Silicon MPS, or CPU)
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# ── Helper functions (used throughout all sections) ──────────────────────────

def make_message(system_content, user_content):
    """Wrap system + user text into the chat-template format."""
    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": user_content},
    ]

def fetch_response(response):
    """Extract the assistant reply text from a pipeline response."""
    return response[0]["generated_text"][-1]["content"]

# ── Demo ─────────────────────────────────────────────────────────────────────

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

system_prompt = "You are a helpful assistant."
user_prompt = "What is the capital of France?"

response = model_pipeline(make_message(system_prompt, user_prompt))
print(fetch_response(response))