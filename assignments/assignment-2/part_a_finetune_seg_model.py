#!/usr/bin/env python3
"""
This script fine-tunes an Ultralytics segmentation model on the Carparts dataset
and saves `best.pt` to a stable models directory.
"""

import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

# Project-relative output locations (stable regardless of current working dir)
assignment_dir = Path(__file__).resolve().parent
runs_dir = assignment_dir / "runs"
models_dir = assignment_dir / "models"
runs_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)

# Use Apple GPU on M-series Macs when available
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Training device: {device}")

# Load a pretrained segmentation model
model = YOLO("yolo26n-seg.pt")

# Memory-safe defaults for MacBook training
results = model.train(
    data="carparts-seg.yaml",
    epochs=10,
    imgsz=512,
    batch=8,
    workers=4,
    amp=False,
    device=device,
    project=str(runs_dir),
    name="part_a_finetune_seg",
    exist_ok=True,
)

# Save best checkpoint to assignments/assignment-2/models/part_a_best.pt
run_dir = Path(results.save_dir)
best_weights_src = run_dir / "weights" / "best.pt"

best_weights_dst = models_dir / "part_a_best.pt"

if not best_weights_src.exists():
    raise FileNotFoundError(f"Could not find best checkpoint at: {best_weights_src}")

shutil.copy2(best_weights_src, best_weights_dst)
print(f"Best weights saved to: {best_weights_dst}")


# NOTE: The following version is adapted from the above to run on Lightning.AI studio
# to take advantage of GPU acceleration. Needed the A100 in order to run 50 epochs for 
# training after which I downloaded the resulting part_a_best.pt file and saved it to the models 
# directory for use in part_b_retrieval_corpus.py and part_c_semantic_search.py
"""
from pathlib import Path
import os
import shutil
import torch
from ultralytics import YOLO

project_root = Path(os.environ.get("LIGHTNING_WORKSPACE", Path.cwd()))
project_root.mkdir(parents=True, exist_ok=True)

runs_dir = project_root / "runs"
models_dir = project_root / "models"
runs_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)

# Pick device: prefer CUDA if available on Lightning
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training device: {device}")

model = YOLO("yolo26n-seg.pt")

results = model.train(
    data="carparts-seg.yaml",
    epochs=20,
    imgsz=512,
    batch=8,
    workers=4,
    amp=True if device == "cuda" else False,
    device=device,
    project=str(runs_dir),
    name="part_a_finetune_seg",
    exist_ok=True,
)

run_dir = Path(results.save_dir)
best_weights_src = run_dir / "weights" / "best.pt"
best_weights_dst = models_dir / "part_a_best.pt"

if not best_weights_src.exists():
    raise FileNotFoundError(f"Could not find best checkpoint at: {best_weights_src}")

shutil.copy2(best_weights_src, best_weights_dst)
print(f"Best weights saved to: {best_weights_dst}")
"""

# NOTE: On AI usage
# I used the help of AI (ChatGPT via web interface) to help with memory-safe defaults
# for training on a MacBook (I have a M3 Pro chip with 16GB of RAM). 
# I also used it to help adapt the script to run on Lightning.AI studio with GPU acceleration