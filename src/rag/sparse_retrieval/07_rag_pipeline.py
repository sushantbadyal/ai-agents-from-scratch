"""
================================================================================
07_rag_pipeline.py  —  A small, reusable RAG class tying retrieval + generation
================================================================================

WHAT THIS FILE DOES
-------------------
Wraps retrieval (PyTerrier) + generation (Llama) into one class, `PyTerrierRAG`,
with a single `answer(question)` method. Three deliberate choices are baked in:
  1. GROUNDING — the system prompt orders the model to answer ONLY from the
     retrieved context and to say so explicitly when the answer isn't there,
     instead of confidently inventing one.
  2. ATTRIBUTION — each passage is labelled with its [docno] and the model is
     asked to cite the ids it used, so answers are auditable.
  3. SOURCES RETURNED — `answer()` returns the retrieved passages alongside the
     text, so a calling app can display or log them.

The class is retriever-agnostic: hand it the wiki BM25 retriever (file 04) or
the synthetic one (file 06) — it behaves the same.

WHY A HAND-WRITTEN CLASS (AND NOT THE ALTERNATIVES)
---------------------------------------------------
  * WHY a thin class of our own:
      This is a "from scratch" tutorial. Writing the retrieve -> build-prompt ->
      generate loop by hand makes grounding, citation, and source-return
      VISIBLE and easy to reason about — which is the whole learning goal.
  * WHY NOT LangChain / LlamaIndex / Haystack:
      Those frameworks are excellent for shipping fast, but they abstract away
      exactly the steps we're trying to teach, add heavy dependencies, and hide
      prompt/grounding behaviour behind their own defaults. They are the natural
      next step once you understand what this class does by hand.
  * WHY the retriever/generator choices are what they are:
      See the headers of file 03 (engine) and helper_functions.py (model) for
      the full alternative comparisons.

REQUIREMENTS (libraries imported in this file)
----------------------------------------------
    (standard library + local modules; transitively pyterrier, torch, transformers)

WHAT EACH MODULE DOES HERE AND WHY
----------------------------------
  * helper_functions (local) : `chat_llama` — the single call that runs the LLM.
  * 04_bm25_tfidf_retrievers (local) : `clean_query` to normalise input.
  * config (local) : defaults (k, max_new_tokens), logging.
================================================================================
"""

from __future__ import annotations                            # Postponed annotations -> cleaner type hints

from typing import Any                                         # Standard library: generic type hints

import config                                                 # Local: defaults, logging, step imports
from helper_functions import chat_llama                       # The one call that runs the Llama generator

log = config.get_logger(__name__)                             # Module-scoped logger

_retr_mod = config.import_step("04_bm25_tfidf_retrievers")    # Reuse clean_query from file 04

# A (docno, score, text) triple describing one retrieved passage.
Passage = tuple[str, float, str]                             # Type alias for readability


class PyTerrierRAG:
    """RAG over a PyTerrier retriever + a transformers chat pipeline.

    WHY THIS EXISTS / WHAT IT SOLVES:
        Files 01-06 built the two halves — a retriever that finds relevant text
        and an LLM that writes prose — but on their own each is incomplete: the
        retriever can't answer in natural language, and the LLM makes things up.
        This class is the bridge: it feeds retrieved passages into the model under
        strict rules (answer only from the context, cite your sources, admit when
        you don't know), turning "search results" into a trustworthy, attributed
        answer. It's the reusable unit an application would actually call.
    """

    # System prompt that forces grounded, citable, no-hallucination answers.
    SYSTEM = (
        "You are a precise question-answering assistant. Answer the user's question "  # Role
        "using ONLY the numbered context passages provided. If the answer is not in "  # Grounding rule
        "the context, reply exactly: 'I don't know based on the provided context.' "   # Refuse instead of guessing
        "Cite the passages you used by their [id] tags."                               # Demand attribution
    )

    def __init__(
        self,
        retriever,
        docno2text: dict[str, str],
        k: int = config.DEFAULT_K,
        max_new_tokens: int = config.DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        self.retriever = retriever                            # The PyTerrier retriever (BM25 / TF-IDF)
        self.docno2text = docno2text                          # id -> passage text fallback lookup
        self.k = k                                            # How many passages to retrieve per question
        self.max_new_tokens = max_new_tokens                  # Cap on the generated answer length

    def retrieve(self, question: str) -> list[Passage]:
        """Return the top-k (docno, score, text) tuples for a question.

        WHY THIS EXISTS / WHAT IT SOLVES:
            The "R" in RAG. It narrows the whole corpus down to the handful of
            passages most likely to contain the answer, cleaning the question
            first and carrying each passage's id + text together so the next step
            can both quote the text and cite where it came from.

        TODO — missing a re-ranker:
            This hands BM25's raw top-k straight to the LLM with no re-ranking
            pass. BM25 is purely lexical (term overlap only), so its rank-1 hit
            isn't always the truly most relevant passage — just the strongest
            keyword match; it can't tell "commander in chief" relates to
            "president." The standard fix is a two-stage pipeline: retrieve a
            WIDER candidate set here (e.g. top 50 via a larger `self.k`), then
            re-score just those with a cross-encoder (e.g. a
            `sentence-transformers` cross-encoder, which scores (query, passage)
            pairs jointly and so can actually judge meaning, not just overlap —
            too slow to run over the whole corpus, but cheap over ~50
            candidates), and only pass ITS top-k on to `build_prompt()`. Not
            implemented yet — until it is, retrieval quality is capped at
            whatever BM25 alone can rank correctly.
        """
        res = self.retriever.search(_retr_mod.clean_query(question)).head(self.k)  # Clean, search, keep top-k
        out: list[Passage] = []                               # Accumulator for retrieved passages
        for row in res.itertuples():                          # Walk each result row
            # Prefer the text stored in the index; fall back to our lookup dict.
            text = getattr(row, "text", None) or self.docno2text.get(row.docno, "")  # Recover passage text
            out.append((row.docno, float(row.score), text))   # Keep id, score, and text together
        return out                                            # List of (docno, score, text)

    def build_prompt(self, question: str, passages: list[Passage]) -> str:
        """Assemble the numbered context + question into one user prompt.

        WHY THIS EXISTS / WHAT IT SOLVES:
            The model can only "know" what's in its prompt, so this is where
            retrieval and generation actually meet. Labelling each passage with
            its [id] does double duty: it gives the model discrete, citable
            evidence, and it lets the reader later verify which source each claim
            came from — the mechanism that makes the answer auditable.
        """
        context = "\n\n".join(                                # Join passages with blank lines between them
            f"[{docno}] {text}"                               # Label each passage with its id for citation
            for docno, _score, text in passages               # ...over every retrieved passage
        )
        return (                                              # The final user-turn prompt
            f"Context passages:\n{context}\n\n"               # The retrieved evidence
            f"Question: {question}\n\n"                       # The actual question
            f"Answer (cite the [id] passages you used):"      # Ask for a cited answer
        )

    def answer(self, question: str, verbose: bool = True) -> dict[str, Any]:
        """Retrieve, prompt, generate — and return the answer plus its sources.

        WHY THIS EXISTS / WHAT IT SOLVES:
            The one public method a caller needs: it runs the full retrieve ->
            build-prompt -> generate chain end to end. Crucially it returns the
            source passages ALONGSIDE the answer, so an application can show its
            citations (or a developer can debug a bad answer by inspecting what
            was actually retrieved) instead of trusting an opaque blob of text.
        """
        passages = self.retrieve(question)                    # 1) Retrieve the top-k passages
        if not passages:                                      # Defensive: retrieval could return nothing
            log.warning("No passages retrieved for question: %s", question)  # Warn so it's not silent
        prompt = self.build_prompt(question, passages)        # 2) Build the grounded prompt
        text = chat_llama(                                    # 3) Generate the answer with Llama
            prompt,                                           # The context + question prompt
            system_prompt=self.SYSTEM,                        # The grounding / citation rules
            max_new_tokens=self.max_new_tokens,               # Answer-length cap
        )
        if verbose:                                           # Optional readable trace
            print(f"Q: {question}")                           # The question
            print("Retrieved:", ", ".join(f"[{d}]" for d, _s, _t in passages))  # Which passage ids were used
            print("A:", text, "\n")                           # The model's answer
        return {"question": question, "answer": text, "sources": passages}      # Structured result for callers


def main() -> None:
    """Wire the synthetic retriever (file 06) into the RAG class and ask a question.

    WHY THIS EXISTS / WHAT IT SOLVES:
        Shows the assembled pipeline answering a question whose answer lives only
        in the invented corpus — the first moment all the pieces run together and
        produce a grounded, cited answer the base model could never have known.
    """
    synth_mod = config.import_step("06_synthetic_corpus_index")  # Load the synthetic retriever builder
    bm25, docno2text = synth_mod.build_synthetic_index()         # Build the synthetic index + retriever

    rag = PyTerrierRAG(bm25, docno2text, k=5)                    # Assemble the RAG pipeline
    log.info("PyTerrierRAG ready over the synthetic corpus.")    # Confirm readiness
    rag.answer("What is Aether-9 and what is it known for?")     # Ask a grounded question (answer in p1)


if __name__ == "__main__":                                    # Only run when executed directly
    main()                                                    # Entry point
