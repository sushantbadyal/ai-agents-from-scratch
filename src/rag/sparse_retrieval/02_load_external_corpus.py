"""
================================================================================
02_load_external_corpus.py  —  Bring in a real (external) corpus to search
================================================================================

WHAT THIS FILE DOES
-------------------
The toy corpus in file 01 had 5 documents; real retrieval runs over thousands.
Here we load `rag-datasets/rag-mini-wikipedia` from the Hugging Face Hub. It
ships two configs:
  * "text-corpus"     -> ~3.2k Wikipedia passages, fields: (passage, id)
  * "question-answer" -> ~900 questions with gold answers: (question, answer, id)

We load both, reshape the passages into the `{docno, text}` structure the
search engine (PyTerrier, next files) expects, and build a `docno2text` lookup
so a retrieved id can be turned back into passage text for the LLM.

WHY THIS APPROACH (AND NOT THE ALTERNATIVES)
--------------------------------------------
  * WHY Hugging Face `datasets`:
      One call fetches AND locally caches the corpus, memory-maps it via Apache
      Arrow (so 3.2k — or millions of — rows don't all sit in RAM), and gives
      reproducible named splits. It is the de-facto standard for sharing corpora.
  * WHY NOT hand-download a CSV/JSON and parse it:
      No caching, no schema, brittle URLs, and you re-invent streaming/memory
      handling that `datasets` already does well.
  * WHY NOT load everything into a database here:
      A DB is the right home for mutable, queried application data; this corpus
      is static, read-only, and consumed once to build an index — a DB would be
      overhead with no payoff at this stage.

REQUIREMENTS (libraries imported in this file)
----------------------------------------------
    datasets

WHAT EACH LIBRARY DOES HERE AND WHY
-----------------------------------
  * datasets : `load_dataset` downloads, caches, and splits the mini-wikipedia
               corpus and Q&A pairs with a single call.
  * config (local) : dataset id + logging.
================================================================================
"""

from __future__ import annotations                            # Postponed annotations -> cleaner type hints

from datasets import load_dataset                             # HF loader: downloads + caches datasets by name

import config                                                 # Local: WIKI_DATASET id + logging

log = config.get_logger(__name__)                             # Module-scoped logger


def load_wiki_corpus() -> tuple[list[dict[str, str]], dict[str, str], object]:
    """Download the mini-wikipedia passages + Q&A and return them ready to index.

    WHY THIS EXISTS / WHAT IT SOLVES:
        A retriever is only as useful as the corpus behind it. This function is
        the "get real data in" step: it fetches a genuine (if small) knowledge
        base once, then reshapes it into the ONE format the rest of the pipeline
        agrees on — a list of {docno, text} dicts plus a docno->text lookup. Doing
        that translation here means files 03/04 never have to care where the data
        came from or what the dataset's original column names were.

    Returns:
        wiki_docs  : list of {"docno": <id>, "text": <passage>} dicts.
        docno2text : {<id>: <passage text>} lookup.
        qa_ds      : the question/answer split (for later evaluation).

    Raises:
        RuntimeError: if the dataset cannot be downloaded/loaded.
    """
    log.info("Loading corpus %r from the Hugging Face Hub...", config.WIKI_DATASET)  # Progress note
    try:                                                      # The download can fail (network, renamed dataset)
        corpus_ds = load_dataset(config.WIKI_DATASET, "text-corpus", split="passages")  # ~3.2k passages
        qa_ds = load_dataset(config.WIKI_DATASET, "question-answer", split="test")       # ~900 Q&A pairs
    except Exception as exc:                                  # Convert to a clear, actionable error
        raise RuntimeError(f"Failed to load dataset {config.WIKI_DATASET!r}.") from exc  # Preserve original cause

    log.info("Corpus rows: %d | Q&A rows: %d", len(corpus_ds), len(qa_ds))  # Report sizes
    log.debug("Corpus columns: %s | Q&A columns: %s", corpus_ds.column_names, qa_ds.column_names)  # Field names

    # PyTerrier wants each document as a dict with STRING 'docno' and 'text' fields.
    wiki_docs = [
        {"docno": str(row["id"]), "text": row["passage"]}
        # Rename id->docno (as str), passage->text
        for row in corpus_ds                                  # ...for every passage
    ]
    docno2text = {d["docno"]: d["text"] for d in wiki_docs}   # id -> passage lookup for downstream use

    return wiki_docs, docno2text, qa_ds                       # Hand everything back to the caller


def main() -> None:
    """Download the corpus and print a quick sample as a sanity check.

    WHY THIS EXISTS / WHAT IT SOLVES:
        Running this file on its own confirms the download works and lets you
        eyeball the data (a sample passage, the longest passage, an example Q&A)
        BEFORE you spend time building an index on top of it — catching a broken
        dataset early instead of deep in the pipeline.
    """
    wiki_docs, docno2text, qa_ds = load_wiki_corpus()         # Load everything

    print("\nSample doc :", wiki_docs[0])                     # Show one reshaped document
    longest = max(wiki_docs, key=lambda d: len(d["text"]))    # Find the longest passage (size intuition)
    print("Longest passage:", len(longest["text"]), "chars")  # Print its length
    print("Sample question:", qa_ds[0]["question"])           # Show one example question
    print("Sample answer  :", qa_ds[0]["answer"])             # ...and its gold answer


if __name__ == "__main__":                                    # Only run when executed directly
    main()                                                    # Entry point
