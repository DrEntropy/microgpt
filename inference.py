"""Plain-float forward pass mirroring microgpt.py's gpt(), capturing intermediates."""

import json
import math
import random


def load_model(path="weights.json"):
    with open(path) as f:
        data = json.load(f)
    return data["weights"], data["metadata"]


def tokenize_starter_text(starter_text, meta):
    """Normalize and tokenize optional starter text, validating model constraints."""
    if starter_text is None:
        return "", []
    if not isinstance(starter_text, str):
        raise ValueError("starter_text must be a string")

    normalized = starter_text.strip().lower()
    if not normalized:
        return "", []

    uchars = meta["uchars"]
    block_size = meta["block_size"]
    max_len = block_size - 1  # leave room for leading BOS at position 0
    if len(normalized) > max_len:
        raise ValueError(f"starter_text too long (max {max_len} characters)")

    char_to_id = {ch: i for i, ch in enumerate(uchars)}
    invalid = sorted({ch for ch in normalized if ch not in char_to_id})
    if invalid:
        pretty = ", ".join(repr(ch) for ch in invalid)
        raise ValueError(f"starter_text contains unsupported characters: {pretty}")

    return normalized, [char_to_id[ch] for ch in normalized]


def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax(logits):
    max_val = max(logits)
    exps = [math.exp(v - max_val) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def generate_step(token_id, pos_id, keys, values, weights, meta):
    """Run one forward pass, returning logits and a dict of intermediates."""
    n_layer = meta["n_layer"]
    n_head = meta["n_head"]
    head_dim = meta["head_dim"]

    intermediates = {}

    tok_emb = weights["wte"][token_id]
    pos_emb = weights["wpe"][pos_id]
    combined_emb = [t + p for t, p in zip(tok_emb, pos_emb)]

    intermediates["tok_emb"] = tok_emb
    intermediates["pos_emb"] = pos_emb
    intermediates["combined_emb"] = combined_emb

    x = rmsnorm(combined_emb)

    attn_weights_all = {}  # per head

    for li in range(n_layer):
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, weights[f"layer{li}.attn_wq"])
        k = linear(x, weights[f"layer{li}.attn_wk"])
        v = linear(x, weights[f"layer{li}.attn_wv"])

        intermediates[f"layer{li}_q"] = q
        intermediates[f"layer{li}_k"] = k
        intermediates[f"layer{li}_v"] = v

        keys[li].append(k)
        values[li].append(v)

        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs : hs + head_dim]
            k_h = [ki[hs : hs + head_dim] for ki in keys[li]]
            v_h = [vi[hs : hs + head_dim] for vi in values[li]]
            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                for t in range(len(k_h))
            ]
            aw = softmax(attn_logits)
            attn_weights_all[f"layer{li}_head{h}"] = aw
            head_out = [
                sum(aw[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(head_dim)
            ]
            x_attn.extend(head_out)

        intermediates[f"layer{li}_head_outputs"] = list(x_attn)
        x = linear(x_attn, weights[f"layer{li}.attn_wo"])
        x = [a + b for a, b in zip(x, x_residual)]
        intermediates[f"layer{li}_post_attn"] = list(x)

        x_residual = x
        x = rmsnorm(x)
        x = linear(x, weights[f"layer{li}.mlp_fc1"])
        mlp_relu = [max(0.0, xi) for xi in x]
        intermediates[f"layer{li}_mlp_relu"] = mlp_relu
        x = linear(mlp_relu, weights[f"layer{li}.mlp_fc2"])
        x = [a + b for a, b in zip(x, x_residual)]
        intermediates[f"layer{li}_final_emb"] = list(x)

    logits = linear(x, weights["lm_head"])
    intermediates["logits"] = logits
    intermediates["attn_weights"] = attn_weights_all

    return logits, intermediates


def sample_token(logits, temperature, meta):
    """Apply temperature and sample a token. Returns (token_id, probs)."""
    scaled = [l / temperature for l in logits]
    probs = softmax(scaled)
    token_id = random.choices(range(meta["vocab_size"]), weights=probs)[0]
    return token_id, probs


def prefill_steps(weights, meta, starter_tokens, temperature=0.5):
    """Run forced prefix tokens after BOS and return (steps, token_ids, keys, values)."""
    n_layer = meta["n_layer"]
    BOS = meta["BOS"]
    uchars = meta["uchars"]

    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]
    token_id = BOS
    token_ids = [BOS]
    steps = []

    for pos_id, forced_token in enumerate(starter_tokens):
        logits, intermediates = generate_step(
            token_id, pos_id, keys, values, weights, meta
        )
        probs = softmax([l / temperature for l in logits])

        input_char = "[START]" if token_id == BOS else uchars[token_id]
        output_char = "[END]" if forced_token == BOS else uchars[forced_token]

        steps.append(
            {
                "pos": pos_id,
                "input_token_id": token_id,
                "input_char": input_char,
                "output_token_id": forced_token,
                "output_char": output_char,
                "probs": probs,
                "intermediates": intermediates,
                "forced": True,
            }
        )

        token_ids.append(forced_token)
        token_id = forced_token

    return steps, token_ids, keys, values


def generate_name(weights, meta, temperature=0.5, seed=None, starter_tokens=None):
    """Generate one name, returning list of step dicts."""
    if seed is not None:
        random.seed(seed)

    block_size = meta["block_size"]
    BOS = meta["BOS"]
    uchars = meta["uchars"]
    starter_tokens = starter_tokens or []

    steps, token_ids, keys, values = prefill_steps(
        weights, meta, starter_tokens, temperature=temperature
    )
    token_id = token_ids[-1]

    for pos_id in range(len(starter_tokens), block_size):
        logits, intermediates = generate_step(
            token_id, pos_id, keys, values, weights, meta
        )
        next_token, probs = sample_token(logits, temperature, meta)

        input_char = "[START]" if token_id == BOS else uchars[token_id]
        output_char = "[END]" if next_token == BOS else uchars[next_token]

        steps.append(
            {
                "pos": pos_id,
                "input_token_id": token_id,
                "input_char": input_char,
                "output_token_id": next_token,
                "output_char": output_char,
                "probs": probs,
                "intermediates": intermediates,
                "forced": False,
            }
        )

        if next_token == BOS:
            break
        token_id = next_token

    name = "".join(s["output_char"] for s in steps if s["output_char"] != "[END]")
    return name, steps
