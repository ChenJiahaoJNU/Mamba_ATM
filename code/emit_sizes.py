"""Write the trainable-parameter count of every model to sizes.json, for the size table."""
import json
import os
import sys

import torch

sys.path.insert(0, r".\libs")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model_zoo  # noqa: E402

MODELS = ["mlp", "lstm", "attention", "attention+conv", "dlinear", "patchtst", "itransformer",
          "timesnet", "ssm-only", "gatedfusion", "gatedfusion-guided"]

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "sizes.json"
    sizes = {}
    for name in MODELS:
        torch.manual_seed(0)
        model = model_zoo.build(name, 8, 60)
        sizes[name] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(sizes, f, indent=2)
    print(json.dumps(sizes, indent=2))
    print("written", out)
