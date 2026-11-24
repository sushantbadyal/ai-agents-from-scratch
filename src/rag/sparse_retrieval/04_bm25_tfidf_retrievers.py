"""
================================================================================
04_bm25_tfidf_retrievers.py  —  Query the index with BM25 and TF-IDF
================================================================================

WHAT THIS FILE DOES
-------------------
With an inverted index built (file 03), we RETRIEVE. A retriever scores every
candidate document against a query with a chosen weighting model and returns the
top-k. This file:
  1. Wraps the index in BM25 and TF-IDF retrievers via `pt.terrier.Retriever`.
  2. Adds `clean_query()`: Terrier parses queries in its own query language, so
     raw punctuation (?, ', /, &) can trigger parse errors on real user input;
     stripping to alphanumerics is a robust fix.
  3. Runs a sample query through BM25 and prints the ranked passages.

WHY BM25 (AND WHY ALSO KEEP TF-IDF) — AND THE ALTERNATIVES
----------------------------------------------------------
  * WHY BM25 as the default:
      BM25 improves on TF-IDF with term-frequency SATURATION (the 10th mention
      of a word adds little over the 2nd) and DOCUMENT-LENGTH normalisation (long
      docs don't win just by being long). It is the strongest classic sparse
      model and the standard baseline.
  * WHY still show TF-IDF:
      It is the conceptual parent from file 01; comparing the two on the same
      index makes BM25's refinements concrete. Swapping is a one-word change
      (`wmodel=`), which is exactly why PyTerrier is convenient here.
  * WHY NOT jump straight to dense / neural rerankers:
      Those improve results but add an embedding/cross-encoder model and cost;
      the standard pattern is BM25 FIRST (cheap, high recall) and then an
      optional neural reranker on the top-k — so BM25 is the right foundation.
  * WHY the retriever lives on PyTerrier (not rank_bm25/bm25s): see file 03's
      header for the full engine comparison.

REQUIREMENTS (libraries imported in this file)
----------------------------------------------
    python-terrier      (import name: pyterrier)   + a Java 11+ runtime

WHAT EACH LIBRARY DOES HERE AND WHY
-----------------------------------
  * pyterrier : `pt.terrier.Retriever` scores documents with a named weighting
                model ("BM25" / "TF_IDF") and returns a ranked results DataFrame.
  * re (standard library) : `clean_query` strips punctuation that upsets the
                Terrier query parser.
  * config (local) : defaults (k), logging, step-module import.
================================================================================
"""

from __future__ import annotations                            # Postponed annotations -> cleaner type hints

import re                                                     # Standard library: regex for query cleaning

import pyterrier as pt                                        # PyTerrier: provides the Retriever

import config                                                 # Local: defaults, logging, step imports

log = config.get_logger(__name__)                             # Module-scoped logger

_QUERY_CLEAN = re.compile(r"[^A-Za-z0-9 ]")                  # Precompiled: match anything that's not alnum/space


def clean_query(q: str) -> str:
    """Strip punctuation that can confuse Terrier's query parser.

    WHY THIS EXISTS / WHAT IT SOLVES:
        Terrier interprets a query in its own little query language, where
        characters like ? ' / & have special meaning. Real user questions are
        full of those, so passing them raw can crash the parser or silently skew
        results. Reducing the query to plain words + spaces makes retrieval
        robust to whatever a user actually types.
    """
    return _QUERY_CLEAN.sub(" ", q).strip()                   # Replace non-alphanumerics with spaces, trim ends


def make_retriever(index, wmodel: str = "BM25", num_results: int = config.DEFAULT_K):
    """Wrap an index in a PyTerrier retriever using the given weighting model.

    WHY THIS EXISTS / WHAT IT SOLVES:
        An index is just the searchable data structure; a RETRIEVER is the thing
        that actually scores and ranks documents for a query. Factoring it into
        one helper means switching the ranking formula (BM25 vs TF-IDF vs others)
        or the top-k is a single argument change, and every part of the code
        builds retrievers the same consistent way.

    Args:
        index: a loaded Terrier index.
        wmodel: weighting model name, e.g. "BM25" or "TF_IDF".
        num_results: top-k documents to return per query.

    Returns:
        A `pt.terrier.Retriever` ready to `.search(query)`.
    """
    return pt.terrier.Retriever(                              # Build the retriever over the index
        index,                                                # Index to search
        wmodel=wmodel,                                        # Scoring model (BM25 / TF_IDF / ...)
        num_results=num_results,                              # Top-k per query
        metadata=["docno", "text"],                           # Also return the stored docno + text columns
    )


def build_retrievers(num_results: int = config.DEFAULT_K):
    """Build BM25 and TF-IDF retrievers over the wiki index.

    WHY THIS EXISTS / WHAT IT SOLVES:
        Gives callers both classic sparse rankers over the SAME index in one
        call, so you can compare them head-to-head (BM25's length-normalised
        scoring vs plain TF-IDF) without rebuilding anything — and hands back the
        docno->text lookup + Q&A pairs the downstream RAG pipeline will need.

    Returns:
        (bm25, tfidf, docno2text, qa_ds)
    """
    index_mod = config.import_step("03_build_inverted_index")  # Reuse file 03 to build/load the index
    wiki_index, docno2text, qa_ds = index_mod.build_wiki_index()  # Get the inverted index + helpers

    bm25 = make_retriever(wiki_index, "BM25", num_results)    # Strong default sparse model
    tfidf = make_retriever(wiki_index, "TF_IDF", num_results)  # Classic baseline for comparison
    return bm25, tfidf, docno2text, qa_ds                     # Return both retrievers + helpers


def _print_results(label: str, question: str, results) -> None:
    """Print a ranked results DataFrame's top-5 rows under a labelled header.

    WHY THIS EXISTS / WHAT IT SOLVES:
        BM25 and TF-IDF results are printed in the identical format so they can
        be compared side by side — one shared printer keeps that format from
        drifting between the two call sites.
    """
    print(f"\nTop {label} results for: {question!r}")             # Header
    for row in results.head(5).itertuples():                       # Iterate top-5 rows
        snippet = row.text[:100].replace("\n", " ")                # First 100 chars, single line
        print(f"  rank {row.rank}  docno={row.docno}  score={row.score:.3f}  | {snippet}...")  # Print each hit


def _print_rank_comparison(question: str, bm25_results, tfidf_results) -> None:
    """Print a rank-by-rank BM25 vs TF-IDF table, flagging where the two disagree.

    WHY THIS EXISTS / WHAT IT SOLVES:
        On short, similarly-sized passages, BM25 and TF-IDF often agree on the
        top few hits — BM25's refinements (length normalisation, term-frequency
        saturation) have little to bite on when passages are short and query
        terms rarely repeat within one. Printing two separate top-5 lists hides
        this: they look identical even though the models DO diverge, just lower
        in the ranking. This table lines both rankings up by docno so any
        disagreement — at any rank — is visible at a glance instead of requiring
        a manual diff of two lists.
    """
    bm25_rank = {row.docno: row.rank for row in bm25_results.itertuples()}   # docno -> its BM25 rank
    tfidf_rank = {row.docno: row.rank for row in tfidf_results.itertuples()}  # docno -> its TF-IDF rank
    all_docnos = set(bm25_rank) | set(tfidf_rank)                            # Every doc either model surfaced
    ordered = sorted(all_docnos, key=lambda d: min(bm25_rank.get(d, 999), tfidf_rank.get(d, 999)))  # Best-rank-first

    print(f"\nBM25 vs TF-IDF rank comparison for: {question!r}")             # Header
    print(f"  {'docno':<8} {'bm25_rank':>9} {'tfidf_rank':>10}")             # Column headers
    for docno in ordered:                                                    # One row per doc, best rank first
        br, tr = bm25_rank.get(docno, "-"), tfidf_rank.get(docno, "-")      # "-" if a model didn't return it
        flag = "  <-- differs" if br != tr else ""                          # Flag any disagreement
        print(f"  {docno:<8} {str(br):>9} {str(tr):>10}{flag}")


def main() -> None:
    """Demonstrate live BM25 and TF-IDF searches over the wikipedia index.

    WHY THIS EXISTS / WHAT IT SOLVES:
        Proves the retrieval half of RAG works on its own — take a real question,
        clean it, and show the ranked passages BM25 AND TF-IDF each pull back —
        before we hand those passages to an LLM. Running both (not just BM25)
        is what actually lets you see BM25's refinements (length normalisation,
        term-frequency saturation) change the ranking versus plain TF-IDF. A
        wider top-k than the usual retrieval default is used here so the
        rank-comparison table below has enough rows for a divergence to show up
        in — the top handful of hits often agree between the two models.
    """
    NUM_RESULTS_FOR_DEMO = 15                                       # Wider than config.DEFAULT_K, just for this demo
    bm25, tfidf, _docno2text, _qa_ds = build_retrievers(NUM_RESULTS_FOR_DEMO)  # Build both retrievers

    question = "Who was the first President of the United States?"  # Our own query (differs from the tutorial's)
    cleaned = clean_query(question)                                 # Normalise it for Terrier
    log.info("Raw question: %s | cleaned: %s", question, cleaned)   # Log both forms

    bm25_results = bm25.search(cleaned)                             # Run BM25 -> ranked results DataFrame
    tfidf_results = tfidf.search(cleaned)                           # Run TF-IDF -> ranked results DataFrame

    _print_results("BM25", question, bm25_results)                 # Show BM25's top-5 (with passage text)
    _print_results("TF-IDF", question, tfidf_results)               # Show TF-IDF's top-5, for comparison
    _print_rank_comparison(question, bm25_results, tfidf_results)   # Show WHERE the two rankings actually diverge


if __name__ == "__main__":                                    # Only run when executed directly
    main()                                                    # Entry point
