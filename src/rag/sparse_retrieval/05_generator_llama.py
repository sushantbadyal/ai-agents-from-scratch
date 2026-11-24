"""
================================================================================
05_generator_llama.py  —  The "G" in RAG: the answer-writing LLM
================================================================================

WHAT THIS FILE DOES
-------------------
Retrieval (files 01–04) is only half of RAG. The other half is GENERATION: a
language model that reads the retrieved passages and writes the answer. This
file exercises that generator — Llama-3.2-3B-Instruct — through the thin
`chat_llama()` wrapper in `helper_functions.py`.

Crucially, this file uses NO retrieval yet. It shows the model answering from
its own frozen, parametric memory, which exposes the gap RAG exists to close:
for anything past its training cutoff (or anything private) the model must
guess, hedge, or hallucinate.

WHY THIS APPROACH (AND NOT THE ALTERNATIVES)
--------------------------------------------
  * WHY a local open-weight model via `transformers`:
      Offline, no API key/cost, reproducible, and it teaches the real mechanics.
  * WHY this model (`unsloth/Llama-3.2-3B-Instruct`):
      Laptop-sized, instruction-tuned, and non-gated (no Hub approval needed);
      matches ../../llm_setup.
  * WHY NOT the gated meta-llama build, a hosted API (Claude/OpenAI), or a
    serving stack (vLLM/TGI):
      See the full rationale in `helper_functions.py` — all three are valid
      generator choices; each trades away offline/local simplicity for access
      approval, cost/network, or setup weight.

REQUIREMENTS (libraries imported in this file)
----------------------------------------------
    torch, transformers      (pulled in via helper_functions)

WHAT EACH LIBRARY / MODULE DOES HERE AND WHY
--------------------------------------------
  * helper_functions (local) : `chat_llama` (prompt -> answer) and `load_llm`
                               (cached pipeline). Keeps model plumbing out of
                               this file.
  * config (local) : logging.
================================================================================
"""

from __future__ import annotations                            # Postponed annotations -> cleaner type hints

import config                                                 # Local: logging
from helper_functions import chat_llama, load_llm             # Thin wrappers around the transformers pipeline

log = config.get_logger(__name__)                             # Module-scoped logger


def main() -> None:
    """Ask the bare model a knowable and an un-knowable question, to show the gap.

    WHY THIS EXISTS / WHAT IT SOLVES:
        Motivates the entire tutorial. By asking the model one question it CAN
        answer from training (photosynthesis) and one it CANNOT (a made-up event),
        it makes the core limitation visible: a frozen LLM confidently guesses
        when it lacks the facts. Seeing that failure first is what makes the value
        of retrieval — feeding the model the facts — obvious in the later files.
    """
    load_llm()                                                # Trigger the (cached) model load up front

    # A timeless, general-knowledge question the model can answer from training.
    q1 = "In one sentence, what is the process of photosynthesis?"     # Model should handle this well
    print("Q1:", q1)                                                   # Show the question
    print("A1:", chat_llama(q1), "\n")                                 # Ask Llama and print the answer

    # A made-up recent event -> the model has no reliable knowledge, so it will
    # likely hedge or hallucinate. This is exactly the gap retrieval closes.
    q2 = "Which team won the 2026 Aurora Cup of interplanetary chess?"  # Fictional -> model cannot know
    print("Q2:", q2)                                                    # Show the question
    print("A2:", chat_llama(q2))                                        # Note how unreliable this answer is


if __name__ == "__main__":                                    # Only run when executed directly
    main()                                                    # Entry point
