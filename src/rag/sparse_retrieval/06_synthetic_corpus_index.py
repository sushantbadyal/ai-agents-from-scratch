"""
================================================================================
06_synthetic_corpus_index.py  —  A tiny SYNTHETIC corpus to prove retrieval works
================================================================================

WHAT THIS FILE DOES
-------------------
To demonstrate RAG's value unmistakably, we point the pipeline at a small
SYNTHETIC knowledge base — 15 short paragraphs we invent on deliberately
fictional topics — and build a Terrier index over them (reusing file 03's
`build_or_load_dict_index`), then run a sanity-check query.

*** IMPORTANT: The paragraphs below are FICTIONAL, written purely to illustrate
    retrieval. Do NOT treat any of them as real facts. ***

WHY A SYNTHETIC CORPUS (AND NOT THE ALTERNATIVES)
-------------------------------------------------
  * WHY invented facts:
      The model CANNOT have memorised them, so a correct answer can ONLY have
      come from the retrieved passages — not the model's training data. That
      makes the with-vs-without-retrieval comparison (file 08) airtight.
  * WHY NOT just reuse the real wikipedia corpus for the demo:
      The model already knows a lot of Wikipedia, so a right answer wouldn't
      prove retrieval did the work — it could be recalling training data.
  * WHY NOT a huge synthetic corpus:
      15 self-contained paragraphs keep the signal obvious and the index
      instant to build; scale is already demonstrated on wikipedia (files 02–04).
  * WHY still index it with PyTerrier (not just string-match):
      To exercise the exact same retrieval path the real pipeline uses, so the
      demo reflects real behaviour rather than a shortcut.

REQUIREMENTS (libraries imported in this file)
----------------------------------------------
    python-terrier      (import name: pyterrier)   + a Java 11+ runtime

WHAT EACH LIBRARY / MODULE DOES HERE AND WHY
--------------------------------------------
  * pyterrier : provides the BM25 `Retriever` for the sanity check.
  * 03_build_inverted_index (local) : reuse `build_or_load_dict_index` so the
                indexing logic lives in one place.
  * 04_bm25_tfidf_retrievers (local) : reuse `clean_query` + `make_retriever`.
  * config (local) : index name + logging.
================================================================================
"""

from __future__ import annotations                            # Postponed annotations -> cleaner type hints

import config                                                 # Local: index name, logging, step imports

log = config.get_logger(__name__)                             # Module-scoped logger

# ── 15 FICTIONAL paragraphs (all invented — not real facts) ───────────────────
PARAGRAPHS: list[dict[str, str]] = [
    {"docno": "p1",  "text": "Aether-9 is a fictional multimodal AI model released by the fictional lab Nimbus Systems, capable of reasoning over text, audio, and 3D scenes. It is best known for planning multi-step robotic tasks with almost no fine-tuning."},
    {"docno": "p2",  "text": "The Zephyr Institute announced in 2027 that its researchers had grown coral that thrives in warm, acidic water. The strain, nicknamed Emberreef, is being trialled off the invented coast of Marisail."},
    {"docno": "p3",  "text": "In the fictional 2026 Orbital Premier League final, the Solaris Comets defeated the Vega Titans by nineteen runs at the imaginary Helios Dome. Captain Ravi Anand was named player of the tournament."},
    {"docno": "p4",  "text": "The 2027 Continental Chess Marathon was jointly hosted by the fictional cities of Port Almyra and New Verdant. It was the first event to combine rapid, blitz, and correspondence formats into a single championship."},
    {"docno": "p5",  "text": "Glimmerwood is an invented alpine forest whose leaves emit a faint blue glow at dusk. Botanists in this fiction attribute the glow to a symbiotic fungus called Lumenroot that lives on the tree bark."},
    {"docno": "p6",  "text": "The fictional startup Tidewave Robotics unveiled a self-repairing underwater drone named Coralkeeper. It patrols reef systems, patches damaged sensors on itself, and reports water quality every six hours."},
    {"docno": "p7",  "text": "According to this made-up account, the ancient city of Sunspire was carved entirely from a single ridge of rose quartz. Its central hall is said to amplify whispers into audible speech across a hundred metres."},
    {"docno": "p8",  "text": "The invented spacecraft Meridian-7 completed a fictional flyby of the dwarf planet Coreth in 2029, returning images of nine concentric ice rings. Mission lead Dr. Ilse Marren called the rings 'a frozen fingerprint'."},
    {"docno": "p9",  "text": "Brightloom is a fictional textile that changes colour with the wearer's body temperature. It was designed by the imaginary studio Ketta Weave and first shown at the invented Lyra Design Expo."},
    {"docno": "p10", "text": "In this fiction, the Verdant Accord of 2028 was signed by twelve invented nations to share rainfall data across the Callandra basin. It is regarded as the first treaty to treat weather forecasts as a shared public resource."},
    {"docno": "p11", "text": "The fictional programming language Lumen compiles directly to reversible circuits, letting every computation be run backward to recover its inputs. Its mascot is a two-headed otter named Echo."},
    {"docno": "p12", "text": "Mount Halcyon, an invented volcano on the fictional island of Serenne, last erupted in 1904 and now hosts a lake whose water is a vivid turquoise. Local legend in this fiction says the colour predicts the coming season's rainfall."},
    {"docno": "p13", "text": "The made-up beverage company Firethorn Brews created a tea that stays hot for twelve hours without any external power. The trick, in this fiction, is a mineral called Caldite suspended in the ceramic of the cup."},
    {"docno": "p14", "text": "In the fictional 2030 Skyleague Cup of gliding, the pilot Nadia Okonkwo crossed the invented Strait of Marn without a single motor assist. Her record flight lasted eleven hours and forty minutes."},
    {"docno": "p15", "text": "The imaginary museum of Halden City opened a wing dedicated to 'sounds that no longer exist', including the recreated hum of a fictional 1950s tram. Visitors wear bone-conduction headsets to hear the exhibits."},
]


def build_synthetic_index(force_rebuild: bool = config.FORCE_REBUILD_INDEX):
    """Index the fictional paragraphs and return (bm25_retriever, docno2text).

    WHY THIS EXISTS / WHAT IT SOLVES:
        Gives us a knowledge base the model provably could not have memorised, so
        the with-vs-without-retrieval test (file 08) is airtight — any correct
        answer MUST have come from retrieval. It deliberately reuses the very same
        index builder and retriever factory as the real wiki corpus, so the demo
        travels the identical code path, just over tiny invented data.
    """
    index_mod = config.import_step("03_build_inverted_index")   # Reuse the shared index builder
    retr_mod = config.import_step("04_bm25_tfidf_retrievers")   # Reuse make_retriever

    index = index_mod.build_or_load_dict_index(               # Build/load the tiny synthetic index
        PARAGRAPHS,
        config.SYNTHETIC_INDEX_NAME,                          # Index name from config
        docno_len=10,                                         # Short docnos ("p1".."p15")
        force_rebuild=force_rebuild,                          # Honour caller's rebuild preference
    )
    bm25 = retr_mod.make_retriever(index, "BM25")             # BM25 retriever over the synthetic index
    docno2text = {p["docno"]: p["text"] for p in PARAGRAPHS}  # id -> paragraph lookup
    return bm25, docno2text                                   # Return retriever + lookup


def main() -> None:
    """Build the synthetic index and run a sanity-check query (answer is in p3).

    WHY THIS EXISTS / WHAT IT SOLVES:
        Verifies the retriever can actually find the RIGHT invented paragraph
        before we rely on it in the full pipeline. If "which team won the 2026
        Orbital Premier League?" doesn't surface p3 here, the RAG answer in file
        08 would fail too — so we check the weakest link (retrieval) in isolation.
    """
    log.info("Prepared %d fictional paragraphs.", len(PARAGRAPHS))  # Confirm corpus size
    retr_mod = config.import_step("04_bm25_tfidf_retrievers")       # For clean_query
    bm25, _docno2text = build_synthetic_index()                    # Build the index + retriever

    question = "Which team won the 2026 Orbital Premier League?"    # Our own query; answer lives in p3
    results = bm25.search(retr_mod.clean_query(question))          # Clean + search

    print("\nSanity check question:", question)                    # Show the query
    for row in results.head(3).itertuples():                       # Top-3 hits
        print(f"  docno={row.docno}  score={row.score:.3f}  | {row.text[:80]}...")  # id, score, snippet


if __name__ == "__main__":                                    # Only run when executed directly
    main()                                                    # Entry point
