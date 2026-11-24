"""
================================================================================
visualization_util.py  —  Shared helpers for visualizing sparse retrieval vectors
================================================================================

WHAT THIS FILE DOES
-------------------
Small, reusable helpers for SEEING a sparse vector (TF-IDF, BM25 weights, ...)
instead of just its shape:
  * `plot_sparse_vector_heatmap()` — save a label x vocabulary heatmap PNG.
  * `print_sparse_vector_table()`  — print the vocabulary and each row's exact
    weights as plain `label = [ ... ]` arrays.

Deliberately generic (plain terms + label -> vector pairs, not tied to
TfidfVectorizer) so any file that has a sparse vector over a known vocabulary
can reuse these instead of each writing its own plotting code.

WHAT EACH LIBRARY DOES HERE AND WHY
-----------------------------------
  * numpy      : stack per-row vectors into a single matrix for imshow.
  * matplotlib : renders the heatmap; the "Agg" backend is set explicitly so
                 this works in headless/CI environments with no display.
================================================================================
"""

from __future__ import annotations                            # Postponed annotations -> cleaner type hints

from pathlib import Path                                      # Standard library: filesystem paths as objects
from typing import Sequence                                   # Standard library: generic sequence type hint

import numpy as np                                            # NumPy: stack rows into a matrix
import matplotlib                                             # Plotting
matplotlib.use("Agg")                                          # Headless backend: we only ever save to a file
import matplotlib.pyplot as plt                                # Plotting API


def plot_sparse_vector_heatmap(
    terms: Sequence[str],
    rows: dict[str, np.ndarray],
    out_path: Path,
    title: str = "Sparse vectors over the shared vocabulary",
    highlight_last_row: bool = False,
) -> None:
    """Render a label x vocabulary matrix as a heatmap PNG and save it to disk.

    WHY THIS EXISTS / WHAT IT SOLVES:
        A shape like `(5, 45)` tells you dimensions, not what's actually IN the
        matrix — that each row is mostly zeros, with a handful of non-zero
        weights on the terms that row's document (or query) actually contains.
        A heatmap makes that "sparse" pattern visible at a glance.

    Args:
        terms: vocabulary, in column order.
        rows: ordered mapping of row label -> dense weight vector (len == len(terms)).
        out_path: where to save the PNG (parent directories are created).
        title: plot title.
        highlight_last_row: draw a separator line above the last row — useful
            when the last row is a query being compared against the rest.
    """
    matrix = np.vstack(list(rows.values()))                   # Stack rows into one (n_rows x n_terms) matrix
    row_labels = list(rows.keys())                            # Row labels, in the same order as the matrix

    fig, ax = plt.subplots(figsize=(max(8, len(terms) * 0.35), max(3, len(row_labels) * 0.5)))  # Size to content
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")     # Cell brightness = weight
    ax.set_xticks(range(len(terms)))                          # One tick per vocabulary term
    ax.set_xticklabels(terms, rotation=90, fontsize=8)        # Vertical labels so words don't overlap
    ax.set_yticks(range(len(row_labels)))                     # One tick per row
    ax.set_yticklabels(row_labels)
    if highlight_last_row and len(row_labels) > 1:            # Optionally mark off the last row (e.g. a query)
        ax.axhline(len(row_labels) - 1.5, color="red", linewidth=1.5)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="weight")                   # Scale reference for the color mapping
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)        # Ensure the output folder exists
    fig.savefig(out_path, dpi=150)                            # Write the PNG
    plt.close(fig)                                            # Free the figure (avoids leaking memory in a loop)


def print_sparse_vector_table(
    terms: Sequence[str],
    rows: dict[str, np.ndarray],
    precision: int = 2,
) -> None:
    """Print the vocabulary and each row's weights as plain `label = [ ... ]` arrays.

    WHY THIS EXISTS / WHAT IT SOLVES:
        The heatmap shows the sparsity pattern at a glance, but sometimes you
        want the exact numbers behind it — e.g. to confirm two documents that
        look similarly bright on a term actually have different weights. This
        prints the same rows the heatmap plots, positionally aligned with the
        vocabulary, so `vocabulary[i]` lines up with `rows[label][i]` in every
        printed line:

            vocabulary = ['cat', 'dog', 'in', 'the']
            doc0       = [0.0, 0.58, 0.58, 0.58]
            doc1       = [0.58, 0.0, 0.58, 0.0]

    Args:
        terms: vocabulary, in column order.
        rows: ordered mapping of row label -> dense weight vector (len == len(terms)).
        precision: decimal places to round each weight to.
    """
    label_width = max((len(label) for label in rows), default=0)  # Pad labels so '=' signs line up
    print(f"{'vocabulary':<{label_width}} = {list(terms)}")   # Column headers, in the same order as every row
    for label, vec in rows.items():                           # One line per row, positionally aligned
        rounded = [round(float(w), precision) for w in vec]   # Round for readability, keep zeros explicit
        print(f"{label:<{label_width}} = {rounded}")
