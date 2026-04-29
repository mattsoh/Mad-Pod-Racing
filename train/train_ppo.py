import argparse
import csv
import os
import numpy as np
import torch
from env.mad_pod_env import VectorEnv, MadPodEnv, OBS_SIZE, ACTION_SIZE
from sim.replay import ReplayRecorder
from train.ppo import PPOAgent, PolicyNet


def policy_action_mean(policy, obs, device):
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    mean, _ = policy(obs_t)
    action = torch.tanh(mean)
    return action.detach().cpu().numpy()


def run_eval(agent, opponent, env, device, max_steps=600, opponent_mode="snapshot", baseline_prob=0.5, rng=None):
    obs0, obs1 = env.reset()
    recorder = ReplayRecorder(env.game.checkpoints, env.laps)
    done = False
    total_reward = 0.0
    steps = 0
    info = {}

    while not done and steps < max_steps:
        action0, _, _ = agent.act(obs0, deterministic=True)
        if opponent_mode == "baseline":
            action1 = env.baseline_actions(1)
        elif opponent_mode == "mix":
            use_baseline = False
            if rng is not None:
                use_baseline = rng.random() < baseline_prob
            if use_baseline:
                action1 = env.baseline_actions(1)
            else:
                action1 = policy_action_mean(opponent, obs1, device)
        else:
            action1 = policy_action_mean(opponent, obs1, device)
        obs0, obs1, r0, _, done, info = env.step(action0, action1)
        total_reward += r0
        recorder.add_frame(env.game)
        steps += 1

    return total_reward, info.get("winner"), recorder


def run_eval_suite(agent, opponent, env, device, episodes, opponent_mode, baseline_prob, rng):
    wins = 0
    total_return = 0.0
    last_recorder = None
    for _ in range(episodes):
        eval_return, winner, recorder = run_eval(
            agent,
            opponent,
            env,
            device,
            opponent_mode=opponent_mode,
            baseline_prob=baseline_prob,
            rng=rng,
        )
        total_return += eval_return
        if winner == 0:
            wins += 1
        last_recorder = recorder
    avg_return = total_return / max(episodes, 1)
    winrate = wins / max(episodes, 1)
    return avg_return, winrate, last_recorder


def curriculum_params(progress):
    if progress < 0.2:
        return {"randomize_track": False, "checkpoint_count": 3}
    if progress < 0.5:
        return {"randomize_track": False, "checkpoint_count": 4}
    if progress < 0.8:
        return {"randomize_track": True, "checkpoint_min": 4, "checkpoint_max": 4}
    return {"randomize_track": True, "checkpoint_min": 2, "checkpoint_max": 8}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=2000000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--opponent-update-interval", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--baseline-mix-start", type=float, default=0.6)
    parser.add_argument("--baseline-mix-end", type=float, default=0.2)
    parser.add_argument("--eval-opponent", type=str, default="mix", choices=["snapshot", "baseline", "mix"])
    parser.add_argument("--eval-baseline-prob", type=float, default=0.5)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-checkpoint-min", type=int, default=2)
    parser.add_argument("--eval-checkpoint-max", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-path", type=str, default="runs/agent_latest.pt")
    parser.add_argument("--best-metric", type=str, default="eval_return", choices=["eval_return", "eval_winrate"])
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    os.makedirs("runs", exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    mix_rng = np.random.default_rng(args.seed + 999)
    eval_rng = np.random.default_rng(args.seed + 4242)

    venv = VectorEnv(
        args.num_envs,
        laps=3,
        checkpoint_count=4,
        randomize_track=True,
        checkpoint_min=2,
        checkpoint_max=8,
        seed=args.seed,
    )

    agent = PPOAgent(OBS_SIZE, ACTION_SIZE, args.hidden_size, args.lr, device)
    opponent = PolicyNet(OBS_SIZE, ACTION_SIZE, args.hidden_size).to(device)

    if args.resume:
        best_path = os.path.join("runs", "agent_best.pt")
        resume_path = best_path if os.path.exists(best_path) else args.resume_path
        if os.path.exists(resume_path):
            state = torch.load(resume_path, map_location=device)
            agent.policy.load_state_dict(state)
    opponent.load_state_dict(agent.policy.state_dict())
    opponent.eval()

    eval_env = MadPodEnv(
        laps=3,
        checkpoint_count=4,
        randomize_track=True,
        checkpoint_min=2,
        checkpoint_max=8,
        seed=args.seed + 1234,
    )

    obs0, obs1 = venv.reset()
    episode_returns = np.zeros(args.num_envs, dtype=np.float32)

    num_updates = args.total_steps // (args.num_envs * args.rollout_len)
    best_eval_return = float("-inf")
    best_eval_winrate = float("-inf")

    header = ["update", "mean_return", "eval_return", "eval_winrate"]
    metrics_path = os.path.join("runs", "metrics.csv")
    write_header = not os.path.exists(metrics_path)
    if not write_header:
        with open(metrics_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        if first_line and first_line != ",".join(header):
            metrics_path = os.path.join("runs", "metrics_v2.csv")
            write_header = True
    with open(metrics_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(header)

        for update in range(1, num_updates + 1):
            progress = update / max(num_updates, 1)
            if args.curriculum:
                params = curriculum_params(progress)
                venv.set_track_params(**params)
                eval_env.set_track_params(**params)

            buffer = PPOAgentBuffer(args.rollout_len, args.num_envs)
            for t in range(args.rollout_len):
                action0, logprob0, value0 = agent.act(obs0, deterministic=False)
                with torch.no_grad():
                    action1 = policy_action_mean(opponent, obs1, device)

                baseline_prob = args.baseline_mix_end + (args.baseline_mix_start - args.baseline_mix_end) * (1.0 - progress)
                if baseline_prob > 0.0:
                    mask = mix_rng.random(args.num_envs) < baseline_prob
                    if np.any(mask):
                        baseline_actions = np.stack(
                            [venv.envs[i].baseline_actions(1) for i in range(args.num_envs)]
                        )
                        action1[mask] = baseline_actions[mask]

                next_obs0, next_obs1, r0, _, dones, infos = venv.step(action0, action1)

                buffer.add(t, obs0, action0, logprob0, r0, dones.astype(np.float32), value0)

                episode_returns += r0
                for i, done in enumerate(dones):
                    if done:
                        episode_returns[i] = 0.0

                obs0, obs1 = next_obs0, next_obs1

            with torch.no_grad():
                _, _, last_value = agent.act(obs0, deterministic=True)
            buffer.compute_returns(last_value, args.gamma, args.gae_lambda)

            agent.update(
                buffer,
                clip=args.clip,
                epochs=args.epochs,
                minibatch_size=args.minibatch_size,
                vf_coef=args.vf_coef,
                ent_coef=args.ent_coef,
                max_grad_norm=args.max_grad_norm,
            )

            if update % args.opponent_update_interval == 0:
                opponent.load_state_dict(agent.policy.state_dict())
                opponent.eval()

            if update % args.save_interval == 0:
                torch.save(agent.policy.state_dict(), os.path.join("runs", "agent_latest.pt"))

            eval_return = ""
            eval_winrate = ""
            if update % args.eval_interval == 0:
                eval_env.set_track_params(
                    randomize_track=True,
                    checkpoint_min=args.eval_checkpoint_min,
                    checkpoint_max=args.eval_checkpoint_max,
                )
                eval_return, eval_winrate, recorder = run_eval_suite(
                    agent,
                    opponent,
                    eval_env,
                    device,
                    episodes=args.eval_episodes,
                    opponent_mode=args.eval_opponent,
                    baseline_prob=args.eval_baseline_prob,
                    rng=eval_rng,
                )
                if recorder is not None:
                    recorder.to_json(os.path.join("runs", "last_replay.json"))

                if args.best_metric == "eval_return":
                    improved = eval_return > best_eval_return
                else:
                    improved = eval_winrate > best_eval_winrate

                if improved:
                    best_eval_return = eval_return
                    best_eval_winrate = eval_winrate
                    torch.save(agent.policy.state_dict(), os.path.join("runs", "agent_best.pt"))
                    if recorder is not None:
                        recorder.to_json(os.path.join("runs", "best_replay.json"))

            mean_return = float(np.mean(episode_returns))
            writer.writerow([update, mean_return, eval_return, eval_winrate])
            csvfile.flush()

            if update % args.log_interval == 0:
                pct = 100.0 * update / max(num_updates, 1)
                eval_str = ""
                if eval_return != "":
                    eval_str = f", eval_return={eval_return:.2f}, winrate={eval_winrate:.2f}"
                print(
                    f"update {update}/{num_updates} ({pct:.1f}%) mean_return={mean_return:.2f}"
                    f"{eval_str}",
                    flush=True,
                )


class PPOAgentBuffer:
    def __init__(self, rollout_len, num_envs):
        from train.ppo import RolloutBuffer

        self.buffer = RolloutBuffer.create(rollout_len, num_envs, OBS_SIZE, ACTION_SIZE)

    def add(self, t, obs, actions, logprobs, rewards, dones, values):
        self.buffer.add(t, obs, actions, logprobs, rewards, dones, values)

    def compute_returns(self, last_value, gamma, gae_lambda):
        self.buffer.compute_returns(last_value, gamma, gae_lambda)

    def __getattr__(self, name):
        return getattr(self.buffer, name)


if __name__ == "__main__":
    main()
