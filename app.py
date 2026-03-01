"""Flask app for microgpt educational UI."""

import random

from flask import Flask, jsonify, render_template, request

from inference import generate_name, generate_step, load_model, sample_token

app = Flask(__name__)
weights, meta = load_model()


@app.route("/")
def index():
    return render_template("index.html", meta=meta)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.get_json(force=True, silent=True) or {}
    temperature = float(data.get("temperature", 0.5))
    temperature = max(0.01, min(2.0, temperature))
    seed = data.get("seed", random.randint(0, 2**31))

    name, steps = generate_name(weights, meta, temperature=temperature, seed=seed)
    return jsonify({"name": name, "steps": steps, "seed": seed})


@app.route("/api/step", methods=["POST"])
def api_step():
    """Stateless single-step: client sends token history, server replays and computes one step."""
    data = request.get_json(force=True)
    token_ids = data["token_ids"]  # list of token IDs so far (starting with BOS)
    temperature = float(data.get("temperature", 0.5))
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
