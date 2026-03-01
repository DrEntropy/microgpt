# microgpt

An educational web UI that lets you see inside a tiny transformer as it generates names, character by character.

Built on top of Andrej Karpathy's [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) — a ~200-line pure-Python character-level GPT that implements everything from scratch (autograd, attention, training) with zero dependencies. See his [blog post](https://karpathy.github.io/2026/02/12/microgpt/) for more.

## What it shows

The UI visualizes each step of name generation through 6 pipeline cards:

1. **Tokenization** — how characters map to token IDs (27 tokens: a-z + \[START\])
2. **Embedding** — token and position embeddings combined into a 16-number vector
3. **Attention** — heatmaps for each of the 4 attention heads showing which earlier characters the model focuses on
4. **Combining** — per-head outputs (4 groups of 4), blended via `wo`, then refined through an MLP (64 neurons) into a final 16-dim embedding
5. **Projection** — the final embedding is projected through `lm_head` into 27 raw logits (one per character), with the top 5 labeled
6. **Prediction** — probability bar chart over all possible next characters (softmax of logits)

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync

# Train the model and save weights (~1 min)
uv run python save_weights.py

# Start the web UI
uv run python app.py
```

Then visit [http://localhost:5001](http://localhost:5001).

## Usage

- **Generate a Name** — generates a full name, animating one character at a time. Click any character to inspect that step's internals.
- **Step Through** — generates one character at a time. Click "Next Character" to advance. Better for deep exploration.
- **Temperature slider** — controls randomness. Low = predictable, high = creative. Recomputes probabilities client-side from raw logits (no server call).

## How it works

`microgpt.py` is the original training script (untouched). `save_weights.py` runs training and serializes the ~4,192 learned parameters to `weights.json`. `inference.py` reimplements the forward pass with plain floats (no autograd) and captures intermediates at every layer. `app.py` serves a Flask API that the single-page frontend calls to generate names and retrieve visualization data.
