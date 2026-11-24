"""
================================================================================
helper_functions.py  —  The generation layer (LLM plumbing)
================================================================================

WHAT THIS FILE DOES
-------------------
This module owns everything about the GENERATOR (the LLM) so the rest of the
package never touches transformers directly:

  * `device`          -> auto-detects CUDA / Apple-Silicon MPS / CPU.
  * `load_llm()`      -> lazily loads Llama-3.2-3B as a Hugging Face
                         text-generation pipeline, CACHED so the ~6 GB of
                         weights load exactly once per process.
  * `make_message()`  -> wraps system + user text into the chat-template format.
  * `fetch_response()`-> pulls the assistant reply out of the raw pipeline output.
  * `chat_llama()`    -> one call: prompt in, answer string out.

WHY THIS APPROACH (AND NOT THE ALTERNATIVES)
--------------------------------------------
  * WHY a local open-weight model via `transformers`:
      Fully offline, no API key or per-token cost, reproducible, and it teaches
      the actual mechanics (tokenizer -> model -> decode) rather than hiding
      them behind a web request. This is a "from scratch" tutorial, so local is
      the point.
  * WHY this specific model (`unsloth/Llama-3.2-3B-Instruct`):
      Small enough to run on a laptop/single GPU, instruction-tuned for chat,
      and NON-GATED — so there is no Hugging Face license approval to click
      through. It matches ../../llm_setup so the whole repo speaks to one model.
  * WHY NOT the gated `meta-llama/Llama-3.2-3B-Instruct`:
      Identical weights, but requires requesting access on the Hub first — an
      avoidable speed bump for a tutorial. (Override via RAG_MODEL_ID if you
      have access.)
  * WHY NOT a hosted API (Anthropic Claude, OpenAI, etc.):
      Higher quality and zero local GPU needed, and a hosted API is a very
      common generator choice — but it adds cost, network dependence, and a key,
      none of which suit a local, offline lesson.
  * WHY NOT a dedicated serving stack (vLLM, TGI):
      Those are the RIGHT answer for high-throughput serving (paged attention,
      continuous batching, tensor parallelism). They are heavier to stand up than
      a single-process `pipeline`, so we keep it simple here and note them as the
      scale-up path.

REQUIREMENTS (libraries imported in this file)
----------------------------------------------
    torch
    transformers

WHAT EACH LIBRARY DOES HERE AND WHY
-----------------------------------
  * torch        : PyTorch, the tensor backend the model runs on; used here to
                   detect the best available accelerator (CUDA > MPS > CPU).
  * transformers : Hugging Face library providing `AutoTokenizer`,
                   `AutoModelForCausalLM`, and the high-level `pipeline` that
                   downloads weights, tokenizes, generates, and applies the
                   chat template automatically.
  * config (local) : central model id, logging, and defaults.
================================================================================
"""

from __future__ import annotations                            # Postponed annotations -> cleaner type hints

from typing import Any                                         # Standard library: generic type hints

import torch                                                  # PyTorch: tensor backend + device detection
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline  # HF: tokenizer, model, pipeline wrapper
from transformers.pipelines.base import Pipeline              # Type of the object pipeline() returns

import config                                                 # Local: MODEL_ID, defaults, logging

log = config.get_logger(__name__)                             # Module-scoped logger

# Detect the best available device once, at import time (CUDA GPU > Apple MPS > CPU).
device: str = (
    "cuda" if torch.cuda.is_available()                       # NVIDIA GPU if the CUDA runtime is present
    else "mps" if torch.backends.mps.is_available()           # Else Apple-Silicon Metal Performance Shaders
    else "cpu"                                                # Else plain CPU (works everywhere, slower)
)
log.info("Selected compute device: %s", device)              # Record the chosen device

# Module-level cache so the (large) model is loaded exactly once per process.
_LLM_PIPELINE: Pipeline | None = None                         # Holds the pipeline after the first load


def load_llm(max_new_tokens: int = config.DEFAULT_MAX_NEW_TOKENS) -> Pipeline:
    """Load (once) and return the Llama chat pipeline.

    WHY THIS EXISTS / WHAT IT SOLVES:
        The model is ~6 GB and takes real time to load, but many files need it.
        Loading it on every call — or in every file — would be painfully slow, so
        we build the pipeline once and cache it at module level; every later call
        returns instantly. This is the single choke-point where the whole package
        acquires its generator, so there's one place to change how it's loaded.

    Args:
        max_new_tokens: default generation cap baked into the pipeline.

    Returns:
        A ready-to-call text-generation `Pipeline`.

    Raises:
        RuntimeError: if the model or tokenizer cannot be loaded (e.g. the model
        id is gated and you lack access, or the download failed).
    """
    global _LLM_PIPELINE                                      # We assign to the module-level cache
    if _LLM_PIPELINE is not None:                             # Fast path: already loaded
        return _LLM_PIPELINE                                  # Return the cached pipeline immediately

    log.info("Loading generator model %r (first run downloads ~6 GB)...", config.MODEL_ID)  # Progress note
    try:                                                      # Loading can fail for many reasons; be explicit
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)  # Tokenizer matching the model
        model = AutoModelForCausalLM.from_pretrained(         # The causal-LM (next-token predictor) weights
            config.MODEL_ID,
            device_map="auto",                                # Let transformers place layers on the best device(s)
        )
    except Exception as exc:                                  # Wrap low-level errors in a clear, actionable one
        raise RuntimeError(                                   # Re-raise with context for the caller
            f"Failed to load model {config.MODEL_ID!r}. If it is gated, request "
            f"access on the Hugging Face Hub or set RAG_MODEL_ID to a non-gated build."
        ) from exc                                            # Preserve the original traceback

    _LLM_PIPELINE = pipeline(                                 # Wrap tokenizer + model into a callable pipeline
        "text-generation",                                    # Task: generate text token-by-token
        model=model,                                          # The loaded model
        tokenizer=tokenizer,                                  # The matching tokenizer
        max_new_tokens=max_new_tokens,                        # Default generation length cap
        temperature=0.1,                                      # Low temperature -> focused, near-deterministic output
        top_k=10,                                             # Sample only from the 10 most likely next tokens
        top_p=0.85,                                           # Nucleus sampling: keep the top 85% probability mass
    )
    log.info("Generator model ready.")                        # Confirm the model is loaded
    return _LLM_PIPELINE                                      # Return the freshly built pipeline


def make_message(system_content: str, user_content: str) -> list[dict[str, str]]:
    """Wrap a system + user prompt into the chat-template list-of-dicts format.

    WHY THIS EXISTS / WHAT IT SOLVES:
        Instruction-tuned chat models don't take a plain string — they expect a
        list of role-tagged turns (system = rules, user = the ask), which the
        pipeline then renders with the model's exact chat template. This helper
        hides that structural detail so the rest of the code can think in terms
        of "rules + question" instead of message-dict plumbing.
    """
    return [                                                  # The pipeline expects a list of role/content dicts
        {"role": "system", "content": system_content},        # System turn: behaviour / rules
        {"role": "user", "content": user_content},            # User turn: the actual question / instruction
    ]


def fetch_response(response: list[dict[str, Any]]) -> str:
    """Extract just the assistant's reply text from a raw pipeline response.

    WHY THIS EXISTS / WHAT IT SOLVES:
        The pipeline returns a deeply nested structure (a list, containing the
        full running conversation, whose last turn holds the new reply). Callers
        only want the answer string. Isolating that unwrapping in one place means
        if the output shape ever changes, we fix it here — not in every caller.
    """
    return response[0]["generated_text"][-1]["content"]       # Drill into the nested output to the last message


def chat_llama(
    user_prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    max_new_tokens: int = config.DEFAULT_MAX_NEW_TOKENS,
) -> str:
    """One-shot convenience: send a prompt to Llama and return the answer string.

    WHY THIS EXISTS / WHAT IT SOLVES:
        This is the whole generation layer boiled down to one call: load-or-reuse
        the model, format the messages, run it, and return clean text. Every place
        that needs an answer (the no-retrieval baseline in file 05/08, the RAG
        class in file 07) uses this, so none of them touch transformers directly
        — swap the model or serving backend and every caller keeps working.

    Args:
        user_prompt: the user's question / instruction.
        system_prompt: behaviour instructions for the assistant.
        max_new_tokens: cap on generated tokens for this call.

    Returns:
        The assistant's reply text.
    """
    llm = load_llm(max_new_tokens=max_new_tokens)             # Get the (cached) pipeline
    messages = make_message(system_prompt, user_prompt)       # Build the chat message list
    response = llm(messages, max_new_tokens=max_new_tokens)   # Run generation (per-call length override)
    return fetch_response(response)                           # Return only the assistant's reply text
