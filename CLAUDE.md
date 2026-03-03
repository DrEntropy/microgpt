# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

An educational web UI that visualizes a tiny transformer (character-level GPT) generating names. Built on Andrej Karpathy's microgpt — a ~200-line pure-Python GPT with zero dependencies. The model has ~4,192 parameters: 1 layer, 4 attention heads, 16-dim embeddings, vocab of 27 (a-z + BOS).

## Commands

```bash
uv sync                        # install dependencies
uv run python save_weights.py  # train model → weights.json (~1 min)
uv run python app.py           # start Flask server on localhost:5001
docker build -t microgpt .     # build Docker image (optional)
docker run -p 5001:5001 microgpt  # run with gunicorn (optional)
fly deploy                     # deploy to Fly.io (optional)
```

Only minimal testing is implemented. CI/CD is via GitHub Actions — pushes to `main` auto-deploy to Fly.io (see `.github/workflows/fly-deploy.yml`).

## Architecture

```
microgpt.py (Karpathy's training script, untouched)
    ↓ executed by save_weights.py
weights.json (plain-float weights + metadata)
    ↓ loaded by app.py at startup
app.py (Flask: GET /, POST /api/generate, POST /api/step)
    ↓ calls inference.py for forward passes
inference.py (plain-float forward pass, captures intermediates)
    ↓ intermediates sent as JSON to frontend
templates/index.html + static/app.js + static/style.css (vanilla JS UI)
```

**Key design decisions:**
- `inference.py` reimplements the forward pass without autograd, capturing every intermediate (embeddings, Q/K/V, attention weights, head outputs, MLP activations, logits) for visualization.
- The `/api/step` endpoint is stateless: the client sends the full token history, and the server replays all tokens to rebuild the KV cache. No server-side session state.
- Temperature recomputation happens client-side — the slider applies softmax to raw logits in JS without a server call.
- `META` (model metadata: vocab_size, n_head, head_dim, block_size, uchars, BOS) is injected into the HTML template via `{{ meta | tojson }}` and used throughout `app.js`.

## UI Structure

The frontend is a single-page app with 6 sequential visualization cards:
1. **Tokenization** (green) — character → token ID
2. **Embedding** (blue) — token emb + position emb → combined 16-dim vector
3. **Attention** (purple) — 4 head heatmaps with character labels on axes
4. **Combining** (teal) — head outputs [4|4|4|4] → after blending → MLP hidden (64) → final embedding (16)
5. **Projection** (rose) — final embedding → lm_head → 27 raw logits
6. **Prediction** (orange) — softmax probability bars

Each card color is defined as a CSS variable in `style.css` and applied via `.card-header.<color>` classes with matching `:has()` selectors on the `.card` border.

## Conventions

- Vanilla JS only — no frameworks, no build step, no bundler.
- All visualization rendering is in `app.js` with functions named `render*()` (e.g., `renderEmbedding`, `renderAttention`, `renderProjection`).
- Embedding/logit visualizations use a diverging blue-white-red color scale (`valueColor()`). MLP neurons use a teal intensity scale with gray for dead (zero) neurons.
- Intermediate keys follow the pattern `layer{i}_*` (e.g., `layer0_head_outputs`, `layer0_post_attn`, `layer0_mlp_relu`, `layer0_final_emb`).
