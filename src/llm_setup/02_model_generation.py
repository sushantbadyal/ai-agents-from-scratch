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
# device_map indicates where to load the model parameters
# device_map = None       -> whole model on GPU
# device_map = "auto"     -> spread across multiple GPUs automatically
# device_map = "balanced" -> evenly distributed across GPUs
# device_map = "cpu"      -> run on CPU (slower, but works everywhere)
# Note: device_map is intended for inference, not distributed training

sample_text = "Hello world"
tokens = tokenizer(sample_text, return_tensors="pt").to(device)

output_tokens = model.generate(
    input_ids=tokens["input_ids"],
    attention_mask=tokens["attention_mask"],
    max_new_tokens=100
)
# A cleaner syntax: output_tokens = model.generate(**tokens, max_new_tokens=100)
print("output_tokens:",output_tokens)  # Raw generated token IDs