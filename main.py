import csv
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(ROOT, "runs")


def run_module(module, args):
    cmd = [sys.executable, "-m", module] + args
    print("\nRunning:")
    print(" ".join(cmd))
    print("")
    subprocess.run(cmd, cwd=ROOT, check=False)


def ask_int(prompt, default):
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print("Invalid integer. Using default.")
        return default


def ask_float(prompt, default):
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print("Invalid number. Using default.")
        return default


def ask_str(prompt, default):
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw if raw else default


def best_checkpoint_path():
    best_path = os.path.join(RUNS_DIR, "agent_best.pt")
    if os.path.exists(best_path):
        return best_path
    latest_path = os.path.join(RUNS_DIR, "agent_latest.pt")
    if os.path.exists(latest_path):
        return latest_path
    return ""


def start_training(resume):
    total_steps = ask_int("Total steps", 2_000_000)
    hidden_size = ask_int("Hidden size", 192)
    device = ask_str("Device (auto/cpu/cuda)", "auto")
    eval_episodes = ask_int("Eval episodes", 8)

    args = [
        "--total-steps",
        str(total_steps),
        "--hidden-size",
        str(hidden_size),
        "--curriculum",
        "--baseline-mix-start",
        "0.7",
        "--baseline-mix-end",
        "0.2",
        "--eval-opponent",
        "baseline",
        "--eval-episodes",
        str(eval_episodes),
        "--eval-checkpoint-min",
        "2",
        "--eval-checkpoint-max",
        "8",
        "--device",
        device,
    ]
    if resume:
        args += ["--resume"]
    run_module("train.train_ppo", args)


def start_custom_training():
    total_steps = ask_int("Total steps", 2_000_000)
    hidden_size = ask_int("Hidden size", 128)
    device = ask_str("Device (auto/cpu/cuda)", "auto")
    resume = ask_str("Resume from best? (y/n)", "y").lower().startswith("y")
    curriculum = ask_str("Use curriculum? (y/n)", "y").lower().startswith("y")
    eval_episodes = ask_int("Eval episodes", 5)
    baseline_start = ask_float("Baseline mix start", 0.6)
    baseline_end = ask_float("Baseline mix end", 0.2)

    args = [
        "--total-steps",
        str(total_steps),
        "--hidden-size",
        str(hidden_size),
        "--baseline-mix-start",
        str(baseline_start),
        "--baseline-mix-end",
        str(baseline_end),
        "--eval-episodes",
        str(eval_episodes),
        "--eval-checkpoint-min",
        "2",
        "--eval-checkpoint-max",
        "8",
        "--device",
        device,
    ]
    if resume:
        args += ["--resume"]
    if curriculum:
        args += ["--curriculum"]
    run_module("train.train_ppo", args)


def export_bot(quantized=True):
    checkpoint = best_checkpoint_path()
    if not checkpoint:
        print("No checkpoint found in runs/. Train first.")
        return
    out_path = os.path.join(ROOT, "export", "trained_bot.cpp")
    args = ["--checkpoint", checkpoint, "--out", out_path]
    if not quantized:
        args.append("--no-quantize")
    run_module("export.export_weights", args)
    print(f"Exported: {out_path}")


def view_replay(which):
    replay = os.path.join(RUNS_DIR, "last_replay.json" if which == "last" else "best_replay.json")
    if not os.path.exists(replay):
        print(f"Replay not found: {replay}")
        return
    run_module("viz.viewer", ["--replay", replay])


def summarize_metrics():
    path_v2 = os.path.join(RUNS_DIR, "metrics_v2.csv")
    path_v1 = os.path.join(RUNS_DIR, "metrics.csv")
    path = path_v2 if os.path.exists(path_v2) else path_v1
    if not os.path.exists(path):
        print("No metrics file found.")
        return

    best_return = None
    best_winrate = None
    last_row = None
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            if row.get("update") in (None, "", "update"):
                continue
            last_row = row
            eval_return = row.get("eval_return")
            eval_winrate = row.get("eval_winrate")
            if eval_return not in (None, ""):
                val = float(eval_return)
                best_return = val if best_return is None else max(best_return, val)
            if eval_winrate not in (None, ""):
                val = float(eval_winrate)
                best_winrate = val if best_winrate is None else max(best_winrate, val)

    if not last_row:
        print("Metrics file is empty.")
        return

    print("\nMetrics summary")
    print(f"- file: {path}")
    print(f"- last update: {last_row.get('update', '?')}")
    if last_row.get("mean_return"):
        print(f"- last mean_return: {float(last_row['mean_return']):.2f}")
    if last_row.get("eval_return"):
        print(f"- last eval_return: {float(last_row['eval_return']):.2f}")
    if last_row.get("eval_winrate"):
        print(f"- last eval_winrate: {float(last_row['eval_winrate']):.2f}")
    if best_return is not None:
        print(f"- best eval_return: {best_return:.2f}")
    if best_winrate is not None:
        print(f"- best eval_winrate: {best_winrate:.2f}")


def gpu_status():
    try:
        import torch
    except Exception:
        print("PyTorch not installed. Install torch to check GPU.")
        return
    cuda_ok = torch.cuda.is_available()
    print("\nGPU status")
    print(f"- cuda available: {cuda_ok}")
    if cuda_ok:
        print(f"- device count: {torch.cuda.device_count()}")
        print(f"- device 0: {torch.cuda.get_device_name(0)}")


def main():
    menu = {
        "1": ("Train (recommended, fresh)", lambda: start_training(resume=False)),
        "2": ("Train (recommended, resume best)", lambda: start_training(resume=True)),
        "3": ("Train (custom)", start_custom_training),
        "4": ("Export bot (quantized)", lambda: export_bot(quantized=True)),
        "5": ("Export bot (full precision)", lambda: export_bot(quantized=False)),
        "6": ("View replay (last)", lambda: view_replay("last")),
        "7": ("View replay (best)", lambda: view_replay("best")),
        "8": ("Show metrics summary", summarize_metrics),
        "9": ("GPU status", gpu_status),
        "10": ("Quit", None),
    }

    while True:
        print("\nMad Pod Racing Trainer")
        for key in sorted(menu.keys()):
            print(f"{key}) {menu[key][0]}")
        choice = input("Select an option: ").strip()
        if choice == "10":
            break
        action = menu.get(choice)
        if not action:
            print("Invalid choice.")
            continue
        action[1]()


if __name__ == "__main__":
    main()
