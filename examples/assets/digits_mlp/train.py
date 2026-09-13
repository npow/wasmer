"""Trains the digit classifier used by examples/wasi_nn_gpu_demo.rs and
regenerates model.safetensors / config.json / samples.json in this directory.

Requires: torch, safetensors, scikit-learn, numpy.
Run: python train.py
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import save_file
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0)

X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype(np.float32)  # sklearn digits pixels are 0..16
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

model = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10),
)

opt = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

Xtr = torch.from_numpy(X_train)
ytr = torch.from_numpy(y_train).long()
Xte = torch.from_numpy(X_test)
yte = torch.from_numpy(y_test).long()

for epoch in range(300):
    opt.zero_grad()
    logits = model(Xtr)
    loss = loss_fn(logits, ytr)
    loss.backward()
    opt.step()

model.eval()
with torch.no_grad():
    test_logits = model(Xte)
    test_acc = (test_logits.argmax(dim=1) == yte).float().mean().item()
    train_acc = (model(Xtr).argmax(dim=1) == ytr).float().mean().item()

print(f"train_acc={train_acc:.4f} test_acc={test_acc:.4f} final_loss={loss.item():.4f}")
assert test_acc > 0.85, "model didn't actually learn -- refusing to export a broken demo model"

out_dir = os.path.dirname(os.path.abspath(__file__))

os.makedirs(out_dir, exist_ok=True)

state = model.state_dict()
save_file(state, f"{out_dir}/model.safetensors")

config = {
    "input_dim": 64,
    "layers": [
        {"type": "linear", "weight": "0.weight", "bias": "0.bias"},
        {"type": "relu"},
        {"type": "linear", "weight": "2.weight", "bias": "2.bias"},
    ],
}
with open(f"{out_dir}/config.json", "w") as f:
    json.dump(config, f, indent=2)

# A handful of real test-set samples for the Rust-side demo to run through
# the model and check the predictions against ground truth.
n_samples = 10
sample_idx = np.arange(n_samples)
samples = {
    "inputs": X_test[sample_idx].tolist(),
    "labels": y_test[sample_idx].tolist(),
}
with open(f"{out_dir}/samples.json", "w") as f:
    json.dump(samples, f, indent=2)

print(f"wrote model.safetensors, config.json, samples.json to {out_dir}")
print("tensor names:", list(state.keys()))
for k, v in state.items():
    print(f"  {k}: {tuple(v.shape)}")
