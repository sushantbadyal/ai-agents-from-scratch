"""
================================================================================
03_build_inverted_index.py  —  Turn the corpus into a searchable inverted index
================================================================================

WHAT THIS FILE DOES
-------------------
Cosine similarity over a dense TF-IDF matrix (file 01) is fine for 5 documents
but doesn't scale. Real search engines use an INVERTED INDEX: a map from each
term -> the documents that contain it, plus the statistics (term frequencies,
document lengths) that models like BM25 need.

Here we use PyTerrier — Python bindings over the battle-tested, Java-based
Terrier engine — to build that index from the mini-wikipedia documents (file 02).
Extra touches added on top of the basic flow:
  * INDEX CACHING: if the index already exists on disk we LOAD it instead of
    rebuilding — rebuilding a large corpus on every run is a classic waste.
  * A reusable `build_or_load_dict_index()` helper (also used by file 06).
  * Structured logging + explicit error handling.

WHY PYTERRIER / BM25 (AND NOT THE ALTERNATIVES)  ← read this
------------------------------------------------------------
  * WHY PyTerrier (Terrier engine):
      Builds a real, DISK-BASED inverted index with compressed posting lists, so
      a query only touches documents that contain its terms and the corpus can
      exceed RAM. It exposes classic weighting models (BM25, TF-IDF, PL2, DFR)
      by name, and plugs into learning-to-rank and neural rerankers — i.e. it is
      a real IR stack you can grow into, not a toy.
  * WHY NOT rank_bm25:
      Pure Python, in-RAM, and it re-scores EVERY document for EVERY query in an
      interpreted loop (O(N) per query) with no persistence — it rebuilds on
      each run. Perfect for reading the BM25 formula; wrong for scale.
  * WHY NOT bm25s:
      Fast pure-Python BM25 (precomputed sparse-matrix scores), can persist and
      memory-map — a great Java-free option. But it is still an in-memory-centric
      library without a full IR toolchain (query parsing, LTR, TREC-style
      evaluation), so PyTerrier is the better "learn the real thing" choice.
  * WHY NOT Elasticsearch / OpenSearch:
      THE mainstream engine for large-scale and hybrid search — but it is a
      separate server/cluster to run and operate. Too much moving infrastructure
      for a local, from-scratch tutorial.
  * WHY NOT a vector database (Pinecone/Weaviate/Qdrant/pgvector) with dense
    embeddings:
      Dense retrieval captures paraphrase/meaning that sparse misses, but needs
      an embedding model, is compute-heavier, and is a different paradigm. Sparse
      is the correct first lesson and remains half of modern HYBRID retrieval.
  * COST of PyTerrier: it needs a JVM (Java 11+). That's the one non-pip
    dependency; you already have Java 17, so it's covered.

REQUIREMENTS (libraries imported in this file)
----------------------------------------------
    python-terrier      (import name: pyterrier)   + a Java 11+ runtime

WHAT EACH LIBRARY DOES HERE AND WHY
-----------------------------------
  * pyterrier : `IterDictIndexer` builds the on-disk inverted index from an
                iterable of {docno, text} dicts; `IndexFactory` loads an index
                for querying. Standard, optimised sparse retrieval in Python.
  * config (local) : index paths, logging, and the Terrier JVM starter.
================================================================================
"""

from __future__ import annotations                            # Postponed annotations -> cleaner type hints

from pathlib import Path                                      # Standard library: filesystem paths as objects

import pyterrier as pt                                        # PyTerrier: the sparse-retrieval engine

import config                                                 # Local: paths, logging, JVM starter

log = config.get_logger(__name__)                             # Module-scoped logger


def build_or_load_dict_index(
    docs: list[dict[str, str]],
    index_name: str,
    docno_len: int = 20,
    force_rebuild: bool = config.FORCE_REBUILD_INDEX,
):
    """Build (or load, if it already exists) a Terrier index over {docno, text} dicts.

    WHY THIS EXISTS / WHAT IT SOLVES:
        Scoring a query against every document one-by-one (file 01's approach)
        collapses as the corpus grows. The fix is an INVERTED INDEX — a term ->
        documents map so a query only ever looks at documents that actually
        contain its words. This function is that "make the corpus searchable"
        step. It also solves a practical pain: building the index is slow, so we
        CACHE it to disk and just reload it on later runs instead of rebuilding.
        It's shared (wiki in file 03, synthetic in file 06) so the indexing logic
        lives in exactly one place.

    Args:
        docs: documents as dicts with 'docno' (str) and 'text' fields.
        index_name: sub-folder name under the configured index root.
        docno_len: max stored length of the docno meta field, in characters.
        force_rebuild: rebuild even if a cached index exists.

    Returns:
        A loaded Terrier index object, ready for retrieval.
    """
    config.start_terrier()                                    # Ensure the Terrier JVM is running

    path: Path = config.index_path(index_name)                # Where this index lives on disk
    data_properties = path / "data.properties"                # Terrier writes this file when an index exists

    # Reuse an existing index instead of rebuilding it every run.
    if data_properties.exists() and not force_rebuild:        # Cached index present and no forced rebuild?
        log.info("Loading existing index at %s", path)        # Log the cache hit
        return pt.IndexFactory.of(str(path))                  # Load and return the cached index

    log.info("Building index %r over %d documents at %s", index_name, len(docs), path)  # Log the (re)build
    max_text_len = max((len(d["text"]) for d in docs), default=1) + 1  # Longest text -> stored-text budget (+1 safety)

    indexer = pt.IterDictIndexer(                             # Indexer that consumes {docno, text} dicts
        str(path),                                            # Index location on disk
        meta={"docno": docno_len, "text": max_text_len},      # STORE docno + full text alongside the index
        overwrite=True,                                       # Rebuild cleanly (we only get here when (re)building)
    )
    try:                                                      # Indexing can fail (JVM, disk, malformed docs)
        index_ref = indexer.index(docs)                       # Build the index from the documents
    except Exception as exc:                                  # Surface a clear, actionable error
        raise RuntimeError(f"Failed to build Terrier index {index_name!r} at {path}.") from exc  # Keep cause

    index = pt.IndexFactory.of(index_ref)                     # Load the freshly built index for querying
    log.info("Index stats: %s", index.getCollectionStatistics().toString().replace("\n", " | "))  # Doc/token counts
    return index                                              # Return the ready index


def build_wiki_index(force_rebuild: bool = config.FORCE_REBUILD_INDEX):
    """Build/load the mini-wikipedia index and return (index, docno2text, qa_ds).

    WHY THIS EXISTS / WHAT IT SOLVES:
        A one-call convenience that chains "load the corpus" (file 02) into "index
        it" (the helper above), so every downstream file (04's retrievers, the RAG
        pipeline) can get a ready-to-query wikipedia index with a single call and
        without repeating the load-then-index wiring.
    """
    load_mod = config.import_step("02_load_external_corpus")  # Reuse file 02 to fetch the corpus
    wiki_docs, docno2text, qa_ds = load_mod.load_wiki_corpus()  # Download + reshape the passages

    index = build_or_load_dict_index(                        # Build or load the index over those docs
        wiki_docs,
        config.WIKI_INDEX_NAME,                               # Index name from config
        force_rebuild=force_rebuild,                          # Honour the caller's rebuild preference
    )
    return index, docno2text, qa_ds                          # Return index + helpers


def main() -> None:
    """Build the wikipedia index and print its statistics.

    WHY THIS EXISTS / WHAT IT SOLVES:
        Lets you run the indexing step by itself and confirm it worked — the
        printed collection statistics (document count, token count, vocabulary
        size) are the quickest proof that the corpus was ingested correctly
        before any retrieval is attempted.
    """
    log.info("PyTerrier version: %s", pt.__version__)        # Show which PyTerrier is installed
    build_wiki_index()                                        # Build/load (logs stats as a side effect)


if __name__ == "__main__":                                    # Only run when executed directly
    main()                                                    # Entry point
