"""
================================================================================
08_with_vs_without_retrieval.py  —  The payoff: with retrieval vs. without
================================================================================

WHAT THIS FILE DOES
-------------------
The tutorial's payoff. We ask the SAME questions two ways over the fictional
synthetic corpus (file 06):
  * WITHOUT retrieval -> bare Llama answers from training memory alone. Because
    the facts are invented, it should hedge, refuse, or hallucinate.
  * WITH retrieval (the RAG pipeline from file 07) -> the model is handed the
    relevant fictional passages and should answer correctly and cite them.

Side by side, this makes retrieval's value concrete: retrieval supplies the
fact; the model just phrases it.

WHY THIS COMPARISON (AND NOT THE ALTERNATIVES)
----------------------------------------------
  * WHY test on the fictional corpus:
      It isolates the variable. On real facts a correct "no retrieval" answer
      could just be recalled training data, proving nothing. Invented facts mean
      any correct answer MUST have come from retrieval.
  * WHY a qualitative side-by-side rather than a metric here:
      The goal is intuition — see the failure vs. the grounded success. A
      quantitative evaluation (e.g. exact-match / F1 over the file-02 Q&A pairs,
      or a RAGAS-style faithfulness score) is the natural follow-up, but it
      belongs in a dedicated eval file, not this teaching demo.

REQUIREMENTS (libraries imported in this file)
----------------------------------------------
    (standard library + local modules; transitively pyterrier, torch, transformers)

WHAT EACH MODULE DOES HERE AND WHY
----------------------------------
  * helper_functions (local) : `chat_llama` for the no-retrieval baseline.
  * 06_synthetic_corpus_index (local) : builds the fictional index + retriever.
  * 07_rag_pipeline (local) : the `PyTerrierRAG` class for the with-retrieval run.
  * config (local) : logging + step imports.
================================================================================
"""

from __future__ import annotations                            # Postponed annotations -> cleaner type hints

import config                                                 # Local: logging, step imports
from helper_functions import chat_llama                       # Baseline: model answering with no retrieval

log = config.get_logger(__name__)                             # Module-scoped logger


def with_vs_without(rag, question: str) -> None:
    """Print the model's answer WITHOUT retrieval, then WITH the RAG pipeline.

    WHY THIS EXISTS / WHAT IT SOLVES:
        Turns the tutorial's whole argument into something you can SEE. Asking the
        exact same question both ways — bare model vs. RAG — over facts the model
        can't know isolates retrieval as the single variable, so the difference in
        the two answers is directly attributable to it. This is the "proof", not
        just a claim, that retrieval fixes the knowledge gap from file 05.
    """
    print("=" * 72)                                           # Visual separator
    print("QUESTION:", question)                              # The question under test
    print("-" * 72)                                           # Separator

    # WITHOUT retrieval: the model answers from its own (empty, for these facts) memory.
    no_rag = chat_llama(question, system_prompt="You are a helpful assistant.")  # Bare model call
    print("WITHOUT RETRIEVAL:\n", no_rag)                     # Show the (likely wrong / hedging) answer

    print("-" * 72)                                           # Separator
    # WITH retrieval: the RAG pipeline supplies the fictional passages first.
    print("WITH RETRIEVAL (RAG):")                            # Header
    rag.answer(question)                                      # RAG prints Q / retrieved ids / grounded answer


def main() -> None:
    """Build the RAG pipeline and run the with-vs-without comparison twice.

    WHY THIS EXISTS / WHAT IT SOLVES:
        Runs the comparison on more than one question so the effect reads as a
        reliable pattern, not a lucky single example — the closing demonstration
        that the whole retrieve-then-generate pipeline actually delivers.
    """
    synth_mod = config.import_step("06_synthetic_corpus_index")  # Fictional corpus + retriever
    rag_mod = config.import_step("07_rag_pipeline")             # The PyTerrierRAG class

    bm25, docno2text = synth_mod.build_synthetic_index()       # Build the fictional index + BM25 retriever
    rag = rag_mod.PyTerrierRAG(bm25, docno2text, k=5)          # Assemble the RAG pipeline

    # Two questions whose answers live ONLY in our synthetic corpus.
    with_vs_without(rag, "Which team won the 2026 Orbital Premier League?")   # Answer in paragraph p3
    with_vs_without(rag, "What causes the leaves of Glimmerwood to glow?")     # Answer in paragraph p5


if __name__ == "__main__":                                    # Only run when executed directly
    main()                                                    # Entry point
