"""
================================================================================
01_tfidf_intuition.py  —  What "sparse retrieval" actually means
================================================================================

WHAT THIS FILE DOES
-------------------
Sparse retrieval represents each document as one long vector over the WHOLE
vocabulary, where almost every entry is zero (hence "sparse") — a slot per word,
non-zero only for the words that document actually contains.

This file builds that intuition from scratch on a tiny, readable corpus:
  1. Defines a handful of short documents (`MINI_CORPUS`).
  2. Writes a dead-simple `tokenize()` to split text into lowercase word tokens.
  3. Turns the documents into TF-IDF vectors with scikit-learn's
     `TfidfVectorizer`.
  4. Scores a user query against every document with cosine similarity and
     ranks them — retrieval in its purest form.
  5. Peeks at the learned IDF weights to see WHY rare, informative words
     dominate the score over common ones.

WHY THIS APPROACH (AND NOT THE ALTERNATIVES)
--------------------------------------------
  * WHY scikit-learn here (not PyTerrier, which the rest of the package uses):
      This file's job is INTUITION, not scale. `TfidfVectorizer` exposes the raw
      vectors, the vocabulary, and the IDF weights as plain arrays you can print
      and inspect. PyTerrier would hide that math behind a Java index — the
      wrong tool for a 5-document teaching example.
  * WHY sparse (TF-IDF) and not dense embeddings here:
      Dense embeddings would require loading an embedding model and would hide
      the entire point — that a document is a mostly-zero vector over the
      vocabulary. Sparse is the concept we are here to see.
  * WHY NOT rank_bm25 / bm25s here:
      They compute scores but don't hand you the interpretable vector + IDF
      table as cleanly for teaching; sklearn is the clearest lens for step one.

REQUIREMENTS (libraries imported in this file)
----------------------------------------------
    numpy
    scikit-learn

WHAT EACH LIBRARY DOES HERE AND WHY
-----------------------------------
  * numpy        : fast arrays; used to sort/index similarity scores for ranking.
  * scikit-learn : `TfidfVectorizer` (text -> sparse TF-IDF vectors) and
                   `cosine_similarity` (query-vs-document scoring).
  * config (local) : shared logging setup.
  * utils.visualization_util (local) : shared heatmap + printed-array views of
                sparse vectors, reused across files instead of each plotting
                its own.
================================================================================
"""

from __future__ import annotations                            # Postponed annotations -> cleaner type hints

import re                                                     # Standard library: regex for tokenization

import numpy as np                                            # NumPy: array math + argsort for ranking
from sklearn.feature_extraction.text import TfidfVectorizer   # Turns text into sparse TF-IDF vectors
from sklearn.metrics.pairwise import cosine_similarity        # Measures similarity between vectors

import config                                                 # Local: logging (also puts src/ on sys.path)
from utils.visualization_util import (                        # Shared: heatmap + printed-array views of sparse vectors
    plot_sparse_vector_heatmap,
    print_sparse_vector_table,
)

log = config.get_logger(__name__)                             # Module-scoped logger

# ── A tiny, readable corpus (each string is one "document") ───────────────────
MINI_CORPUS: list[str] = [
    "The blue whale is the largest animal that lives in the ocean.",  # doc 0 — the blue whale
    "Dolphins are intelligent mammals that travel in social pods.",  # doc 1 — dolphins
    "A coral reef is built slowly by tiny creatures called polyps.",  # doc 2 — coral reefs
    "The octopus can change its color to hide from hungry predators.",  # doc 3 — the octopus
    "Sharks have hunted in the sea for hundreds of millions of years.",  # doc 4 — sharks
]

_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")                      # Precompiled splitter: any run of non-alphanumerics


def tokenize(text: str) -> list[str]:
    """Split raw text into comparable word tokens.

    WHY THIS EXISTS / WHAT IT SOLVES:
        Retrieval decides relevance by the WORDS a document and a query share,
        so before we can compare anything we must break free-form text into those
        word units. Lowercasing makes "Whale" and "whale" count as the same term;
        splitting on non-alphanumerics throws away punctuation (which carries no
        matching signal) so "sea." and "sea" also match. (TfidfVectorizer does
        this internally too — we spell it out here so you can SEE the tokens.)
    """
    return [t for t in _TOKEN_SPLIT.split(text.lower()) if t]  # Split, drop empty strings from leading/trailing seps


def rank_by_tfidf(corpus: list[str], query: str) -> list[tuple[int, float]]:
    """Rank the documents by how well each one matches the query.

    WHY THIS EXISTS / WHAT IT SOLVES:
        This IS retrieval in miniature. The problem it solves: a computer can't
        compare a question to documents as raw text, so we turn each document and
        the query into TF-IDF vectors (numbers weighting each word by how
        important it is), then score their overlap with cosine similarity.
        Sorting those scores turns a vague "which document is relevant?" into the
        concrete, ordered list a search engine would hand back — which is exactly
        the job BM25/PyTerrier do at scale in the later files.

    Args:
        corpus: the documents to search.
        query: the natural-language query.

    Returns:
        List of (document index, cosine-similarity score), descending by score.
    """
    vectorizer = TfidfVectorizer()                            # Learns vocabulary + IDF weights
    doc_matrix = vectorizer.fit_transform(corpus)             # TF-IDF vectors for all docs
    log.info("doc_matrix: %s", doc_matrix.shape)
    query_vec = vectorizer.transform([query])                 # TF-IDF vector for the query
    log.info("query_vec: %s", query_vec.shape)
    scores = cosine_similarity(query_vec, doc_matrix)[0]      # cosine sim, computed on those TF-IDF vectors
    log.info("scores: %s", scores.shape)
    order = np.argsort(scores)[::-1]                          # Indices sorted descending (best first)
    log.info("order: %s", order.shape)
    return [(int(i), float(scores[i])) for i in order]        # Pair each doc index with its score


def tfidf_rows(corpus: list[str], query: str) -> tuple[list[str], dict[str, np.ndarray]]:
    """Fit TF-IDF on `corpus` and return (vocabulary, {row_label: dense_vector}), query last.

    WHY THIS EXISTS / WHAT IT SOLVES:
        Both the heatmap and the printed-array view need the exact same terms
        and rows, just rendered differently — this is the one place that builds
        them, so the two views can never drift apart.
    """
    vectorizer = TfidfVectorizer()                            # Fresh vectorizer, just for visualization
    doc_matrix = vectorizer.fit_transform(corpus).toarray()   # Dense (n_docs x vocab) matrix
    query_row = vectorizer.transform([query]).toarray()[0]    # Query mapped into the same vector space
    terms = list(vectorizer.get_feature_names_out())          # Column labels: the shared vocabulary

    rows = {f"doc{i}": doc_matrix[i] for i in range(len(corpus))}  # doc0, doc1, ...
    rows["query"] = query_row                                 # Query goes last, as the row being compared
    return terms, rows


def main() -> None:
    """Run the TF-IDF intuition demo end to end.

    WHY THIS EXISTS / WHAT IT SOLVES:
        Ties the pieces together so you can watch sparse retrieval work on data
        small enough to reason about by eye: rank the corpus against a query, then
        print the IDF weights to reveal WHY rare words (like "octopus") dominate
        the score while common words (like "the") barely count. That intuition is
        the foundation everything else in this folder builds on.
    """
    # Our own query (different from the tutorial's), phrased to test term matching.
    query = "which is the biggest creature living in the sea"          # The question we want documents for
    log.info("Query: %s", query)                                      # Log the query
    log.info("Query tokens: %s", tokenize(query))                     # Show how it tokenizes

    ranking = rank_by_tfidf(MINI_CORPUS, query)                       # Rank the corpus against the query

    terms, rows = tfidf_rows(MINI_CORPUS, query)                      # Same terms/rows feed both views below
    print_sparse_vector_table(terms, rows)                            # See the exact numbers, not just the shape
    heatmap_path = config.VIZ_OUTPUT_DIR / "tfidf_heatmap.png"        # Where to save the visualization
    plot_sparse_vector_heatmap(                                       # See the sparsity pattern at a glance
        terms, rows, heatmap_path,
        title="TF-IDF vectors: each row is a sparse vector over the shared vocabulary",
        highlight_last_row=True,
    )
    log.info("Saved TF-IDF heatmap to %s", heatmap_path)

    print("\n--- TF-IDF ranking (best match first) ---")             # Header for the demo output
    for rank, (doc_idx, score) in enumerate(ranking, start=1):       # Walk ranked docs
        print(f"{rank}. doc{doc_idx}  score={score:.4f}  | {MINI_CORPUS[doc_idx]}")  # rank, id, score, text

    # Peek at IDF weights: rarer words carry more signal.
    vectorizer = TfidfVectorizer().fit(MINI_CORPUS)                   # Fit again just to expose the vocab + IDF
    idf = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))  # word -> IDF weight
    print("\n--- Sample IDF weights (higher = rarer = more informative) ---")  # Header
    for word in ["the", "is", "whale", "coral", "octopus", "sharks"]:  # Compare common vs rare words
        if word in idf:                                              # Only print words present in the vocabulary
            print(f"idf('{word}') = {idf[word]:.3f}")               # 'the' will be low; 'octopus' high


if __name__ == "__main__":                                    # Only run the demo when executed directly
    main()                                                    # Entry point
