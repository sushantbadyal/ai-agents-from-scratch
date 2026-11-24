# ai-agents-from-scratch
AI Agents from Scratch is a hands-on repository covering fundamentals to advanced Generative AI and Agentic AI. It is a practical end-to-end guide to building AI agents from scratch, including core GenAI concepts, agent architectures, tool use, Retrieval-Augmented Generation (RAG), and multi-agent systems.

## Prerequisites

- **Python 3.14** (the version pinned in `pyproject.toml`).
- **Java 11+ JDK** — required by [PyTerrier](https://pyterrier.readthedocs.io/) (used in `src/rag/sparse_retrieval`), which drives the Java-based Terrier search engine via an embedded JVM. Verify with:
  ```bash
  javac -version
  ```
- A GPU or Apple-Silicon machine is recommended for the LLM steps (the Llama-3.2-3B generator downloads ~6 GB of weights on first run), but CPU works too.

## Setup

```bash
# 1. Activate the project virtual environment
source .venv_v1/bin/activate

# 2. Install the project + all dependencies (declared in pyproject.toml)
pip install -e .
```

### Configuration and API keys

API keys (`ANTHROPIC_API_KEY`, `OPEN_AI_API_KEY`, `HF_TOKEN`) and app settings are read from a JSON file that lives **outside** this repository. Point to it with the `APP_CONFIG_PATH` environment variable:

```bash
export APP_CONFIG_PATH="$HOME/path/to/ai-agents-from-scratch.json"
```

If it isn't set, the code looks for `~/.config/ai-agents-from-scratch/config.json`. Never commit this file.

This installs everything the modules need: `torch`, `transformers`, `numpy`, `scikit-learn`, `datasets`, `python-terrier`, and `pydantic`.

## Running the modules

All example modules are plain scripts. Run them from **inside their own directory** so the local `config` / `helper_functions` imports resolve.

### 1. LLM setup (`src/llm_setup`)

Foundations of loading and prompting a local Llama model — run in sequence:

```bash
cd src/llm_setup
python 01_tokenizer.py
python 02_model_generation.py
python 03_chat_template.py
python 04_pipeline_basic.py
python 05_pipeline_chat.py
```

### 2. Sparse retrieval + RAG (`src/rag/sparse_retrieval`)

Sparse retrieval (TF-IDF / BM25 with PyTerrier) wired into a full RAG pipeline — run in sequence:

```bash
cd src/rag/sparse_retrieval
python 01_tfidf_intuition.py            # TF-IDF + cosine similarity on a tiny corpus (CPU-only)
python 02_load_external_corpus.py       # download the rag-mini-wikipedia corpus
python 03_build_inverted_index.py       # build (and cache) the PyTerrier inverted index
python 04_bm25_tfidf_retrievers.py      # query the index with BM25 and TF-IDF
python 05_generator_llama.py            # the Llama generator, answering WITHOUT retrieval
python 06_synthetic_corpus_index.py     # index a small synthetic (fictional) corpus
python 07_rag_pipeline.py               # the reusable PyTerrierRAG class
python 08_with_vs_without_retrieval.py  # the payoff: answers with vs. without retrieval
```

> Files 01–04 are light/CPU-only. Files 05–08 load the Llama weights (~6 GB) on first run.
> The first index build also downloads Terrier's jars and writes the index under a temp dir; later runs reuse the cached index automatically.

#### Configuration (env vars)

The sparse-retrieval package reads all its knobs from `src/rag/sparse_retrieval/config.py`, each overridable via an environment variable:

| Variable | Default | Purpose |
|----------|---------|---------|
| `RAG_MODEL_ID` | `unsloth/Llama-3.2-3B-Instruct` | generator model id |
| `RAG_WIKI_DATASET` | `rag-datasets/rag-mini-wikipedia` | Hugging Face corpus id |
| `RAG_TOP_K` | `5` | passages retrieved per query |
| `RAG_MAX_NEW_TOKENS` | `300` | generated-answer length cap |
| `RAG_INDEX_ROOT` | `<tmpdir>/rag_sparse_retrieval` | where Terrier indexes are stored |
| `RAG_LOG_LEVEL` | `INFO` | logging verbosity |

Example:

```bash
RAG_TOP_K=10 RAG_LOG_LEVEL=DEBUG python 08_with_vs_without_retrieval.py
```

## License

Copyright (c) 2026 Sushant Badyal. **All rights reserved.**

This is not open-source software. Viewing is permitted, but copying, modifying, or distributing this code is prohibited without prior written permission. See [LICENSE](LICENSE) for details.
