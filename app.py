"""Flask app for microgpt educational UI."""

import random

from flask import Flask, jsonify, render_template, request

from inference import (
    generate_name,
    generate_step,
    load_model,
    prefill_steps,
    sample_token,
    tokenize_starter_text,
)

app = Flask(__name__)
weights, meta = load_model()


@app.route("/")
def index():
    return render_template("index.html", meta=meta)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.get_json(force=True, silent=True) or {}
    try:
        temperature = float(data.get("temperature", 0.5))
    except (TypeError, ValueError):
        return jsonify({"error": "temperature must be a number"}), 400
    temperature = max(0.01, min(2.0, temperature))
    seed = data.get("seed", random.randint(0, 2**31))
    try:
        starter_text, starter_tokens = tokenize_starter_text(
            data.get("starter_text", ""), meta
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    name, steps = generate_name(
        weights,
        meta,
        temperature=temperature,
        seed=seed,
        starter_tokens=starter_tokens,
    )
    return jsonify(
        {
            "name": name,
            "steps": steps,
            "seed": seed,
            "starter_text": starter_text,
            "starter_tokens": starter_tokens,
        }
    )


def validate_token_ids(token_ids, meta):
    if not isinstance(token_ids, list) or not token_ids:
        raise ValueError("token_ids must be a non-empty list")
    if len(token_ids) > meta["block_size"]:
        raise ValueError(f"token_ids too long (max {meta['block_size']})")
    if token_ids[0] != meta["BOS"]:
        raise ValueError("token_ids must start with BOS token")

    vocab_size = meta["vocab_size"]
    normalized = []
    for tid in token_ids:
        if not isinstance(tid, int):
            raise ValueError("token_ids must contain integers")
        if tid < 0 or tid >= vocab_size:
            raise ValueError(f"token_ids entries must be in [0, {vocab_size - 1}]")
        normalized.append(tid)
    return normalized


@app.route("/api/step/init", methods=["POST"])
def api_step_init():
    """Initialize step-through mode with optional forced starter text."""
    data = request.get_json(force=True, silent=True) or {}
    try:
        temperature = float(data.get("temperature", 0.5))
    except (TypeError, ValueError):
        return jsonify({"error": "temperature must be a number"}), 400
    temperature = max(0.01, min(2.0, temperature))

    try:
        starter_text, starter_tokens = tokenize_starter_text(
            data.get("starter_text", ""), meta
        )
        steps, token_ids = prefill_steps(
            weights, meta, starter_tokens, temperature=temperature
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    can_continue = len(token_ids) < meta["block_size"]
    return jsonify(
        {
            "steps": steps,
            "token_ids": token_ids,
            "starter_text": starter_text,
            "starter_tokens": starter_tokens,
            "can_continue": can_continue,
        }
    )


@app.route("/api/step", methods=["POST"])
def api_step():
    """Stateless single-step: client sends token history, server replays and computes one step."""
    data = request.get_json(force=True, silent=True) or {}
    try:
        token_ids = validate_token_ids(data.get("token_ids"), meta)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        temperature = float(data.get("temperature", 0.5))
    except (TypeError, ValueError):
        return jsonify({"error": "temperature must be a number"}), 400
    temperature = max(0.01, min(2.0, temperature))

    n_layer = meta["n_layer"]
    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]

    # Replay all previous tokens to rebuild KV cache
    for pos_id, tid in enumerate(token_ids[:-1]):
        generate_step(tid, pos_id, keys, values, weights, meta)

    # Compute the final step with full intermediates
    current_token = token_ids[-1]
    pos_id = len(token_ids) - 1
    logits, intermediates = generate_step(
        current_token, pos_id, keys, values, weights, meta
    )

    seed = data.get("seed", random.randint(0, 2**31))
    random.seed(seed)
    next_token, probs = sample_token(logits, temperature, meta)

    uchars = meta["uchars"]
    BOS = meta["BOS"]
    input_char = "[START]" if current_token == BOS else uchars[current_token]
    output_char = "[END]" if next_token == BOS else uchars[next_token]

    return jsonify(
        {
            "pos": pos_id,
            "input_token_id": current_token,
            "input_char": input_char,
            "output_token_id": next_token,
            "output_char": output_char,
            "probs": probs,
            "intermediates": intermediates,
            "seed": seed,
        }
    )


@app.route("/api/weights")
def api_weights():
    return jsonify(weights)


if __name__ == "__main__":
    app.run(port=5001, debug=True)
