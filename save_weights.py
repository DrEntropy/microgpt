"""Train microgpt and save weights + metadata to weights.json."""

import json

# Run microgpt.py to train the model, capturing its namespace
namespace = {}
with open("microgpt.py") as f:
    exec(f.read(), namespace)

# Extract metadata
uchars = namespace["uchars"]
state_dict = namespace["state_dict"]

metadata = {
    "uchars": uchars,
    "vocab_size": namespace["vocab_size"],
    "n_embd": namespace["n_embd"],
    "n_layer": namespace["n_layer"],
    "n_head": namespace["n_head"],
    "head_dim": namespace["head_dim"],
    "block_size": namespace["block_size"],
    "BOS": namespace["BOS"],
}

# Convert state_dict: Value objects -> plain floats
weights = {}
for name, matrix in state_dict.items():
    weights[name] = [[v.data for v in row] for row in matrix]

data = {"metadata": metadata, "weights": weights}

with open("weights.json", "w") as f:
    json.dump(data, f)

size_kb = len(json.dumps(data)) / 1024
print(f"Saved weights.json ({size_kb:.1f} KB)")
