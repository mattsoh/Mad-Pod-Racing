# Mad Pod Racing Trainer

This project provides:
- A physics-based simulator with 4 pods
- PPO self-play training loop
- Pygame replay viewer
- Export of a standalone C++ bot (no ML deps)

## Setup

1) Create a venv
   python -m venv .venv
2) Activate it (Windows PowerShell)
   .venv\Scripts\Activate.ps1
3) Install deps
   pip install -r requirements.txt

Torch on Windows:
- If pip fails, follow the official install instructions for your CUDA or CPU build.

## Train

python -m train.train_ppo --total-steps 2000000 --curriculum

Useful flags:
- --hidden-size 192 (bigger model during training)
- --baseline-mix-start 0.7 --baseline-mix-end 0.2 (mix baseline opponent)
- --eval-opponent baseline (evaluate vs baseline)
- --eval-episodes 8 --eval-checkpoint-min 2 --eval-checkpoint-max 8 (broader testing)
- --resume --resume-path runs/agent_latest.pt (auto-resume from runs/agent_best.pt if it exists)
- --best-metric eval_winrate (save best model by winrate)
- --device auto (use CUDA if available)

Outputs:
- runs/agent_latest.pt
- runs/metrics.csv
- runs/last_replay.json

## View replay

python -m viz.viewer --replay runs/last_replay.json

## Export a CodinGame bot

python -m export.export_weights --checkpoint runs/agent_latest.pt --out export/trained_bot.cpp

The exporter quantizes weights to reduce code size. Use --no-quantize for full precision.

## Quick menu

python main.py

Notes:
- The simulator is close to CodinGame but not a byte-perfect clone.
- The exported bot runs a small MLP with the same observation encoding as training.
