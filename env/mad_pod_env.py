import math
import numpy as np
from sim.physics import Vec, MAP_W, MAP_H, norm, dot, angle_to, angle_diff
from sim.game import GameState, Command
from env.baseline_bot import BaselineBot

OBS_SIZE = 49
ACTION_SIZE = 8

REWARD_PROGRESS_SCALE = 0.0005
REWARD_CHECKPOINT_BONUS = 1.0
REWARD_SPEED_SCALE = 0.02
REWARD_ANGLE_SCALE = 0.005


def generate_checkpoints(rng, count):
    checkpoints = []
    margin = 1500.0
    min_dist2 = 2500.0 * 2500.0
    for _ in range(count):
        for _ in range(200):
            x = rng.uniform(margin, MAP_W - margin)
            y = rng.uniform(margin, MAP_H - margin)
            if all((x - cp.x) ** 2 + (y - cp.y) ** 2 >= min_dist2 for cp in checkpoints):
                checkpoints.append(Vec(x, y))
                break
        else:
            checkpoints.append(Vec(x, y))
    return checkpoints


class MadPodEnv:
    def __init__(
        self,
        laps=3,
        checkpoint_count=4,
        randomize_track=True,
        checkpoint_min=2,
        checkpoint_max=8,
        seed=None,
    ):
        self.laps = laps
        self.checkpoint_count = checkpoint_count
        self.randomize_track = randomize_track
        self.checkpoint_min = checkpoint_min
        self.checkpoint_max = checkpoint_max
        self.rng = np.random.default_rng(seed)
        self.game = None
        self.last_score = [0.0, 0.0]
        self.last_next_ids = None
        self.baselines = [None, None]
        self.reset()

    def reset(self):
        if self.randomize_track:
            count = int(self.rng.integers(self.checkpoint_min, self.checkpoint_max + 1))
            checkpoints = generate_checkpoints(self.rng, count)
        else:
            checkpoints = generate_checkpoints(self.rng, self.checkpoint_count)
        self.checkpoint_count = len(checkpoints)
        self.game = GameState(self.laps, checkpoints)
        self.baselines = [BaselineBot(checkpoints), BaselineBot(checkpoints)]
        self.last_score[0] = self._score(0)
        self.last_score[1] = self._score(1)
        self.last_next_ids = [
            pod.next_id for team in self.game.teams for pod in team
        ]
        return self._encode(0), self._encode(1)

    def set_track_params(self, randomize_track=None, checkpoint_count=None, checkpoint_min=None, checkpoint_max=None):
        if randomize_track is not None:
            self.randomize_track = randomize_track
        if checkpoint_count is not None:
            self.checkpoint_count = checkpoint_count
        if checkpoint_min is not None:
            self.checkpoint_min = checkpoint_min
        if checkpoint_max is not None:
            self.checkpoint_max = checkpoint_max

    def baseline_actions(self, team_idx):
        return self.baselines[team_idx].act(self.game, team_idx)

    def _score(self, team_idx):
        pods = self.game.teams[team_idx]
        best = pods[0]
        best_prog = best.lap * self.checkpoint_count + best.next_id
        best_dist = norm(self.game.checkpoints[best.next_id] - best.pos)
        for pod in pods[1:]:
            prog = pod.lap * self.checkpoint_count + pod.next_id
            dist = norm(self.game.checkpoints[pod.next_id] - pod.pos)
            if prog > best_prog or (prog == best_prog and dist < best_dist):
                best = pod
                best_prog = prog
                best_dist = dist
        return best_prog * 10000.0 - best_dist

    def _pod_features(self, pod, is_enemy):
        posx = (pod.pos.x - MAP_W * 0.5) / (MAP_W * 0.5)
        posy = (pod.pos.y - MAP_H * 0.5) / (MAP_H * 0.5)
        velx = pod.vel.x / 1000.0
        vely = pod.vel.y / 1000.0
        ang = math.radians(pod.angle)
        ang_cos = math.cos(ang)
        ang_sin = math.sin(ang)
        cp = self.game.checkpoints[pod.next_id]
        relx = (cp.x - pod.pos.x) / MAP_W
        rely = (cp.y - pod.pos.y) / MAP_H
        dist = min(norm(cp - pod.pos) / 20000.0, 1.0)
        lap_frac = pod.lap / max(self.laps, 1)
        next_frac = pod.next_id / max(self.checkpoint_count - 1, 1)
        shield = 0.0 if is_enemy else pod.shield_cd / 3.0
        return [
            posx,
            posy,
            velx,
            vely,
            ang_cos,
            ang_sin,
            relx,
            rely,
            dist,
            lap_frac,
            next_frac,
            shield,
        ]

    def _encode(self, team_idx):
        features = []
        pods = self.game.teams[team_idx] + self.game.teams[1 - team_idx]
        for idx, pod in enumerate(pods):
            features.extend(self._pod_features(pod, idx >= 2))
        features.append(1.0 if self.game.boost_available[team_idx] else 0.0)
        obs = np.asarray(features, dtype=np.float32)
        if obs.shape[0] != OBS_SIZE:
            raise ValueError("OBS_SIZE mismatch")
        return obs

    def _team_motion_metrics(self, team_idx):
        pods = self.game.teams[team_idx]
        best = pods[0]
        best_prog = best.lap * self.checkpoint_count + best.next_id
        best_dist = norm(self.game.checkpoints[best.next_id] - best.pos)
        for pod in pods[1:]:
            prog = pod.lap * self.checkpoint_count + pod.next_id
            dist = norm(self.game.checkpoints[pod.next_id] - pod.pos)
            if prog > best_prog or (prog == best_prog and dist < best_dist):
                best = pod
                best_prog = prog
                best_dist = dist

        cp = self.game.checkpoints[best.next_id]
        to_cp = Vec(cp.x - best.pos.x, cp.y - best.pos.y)
        dist = norm(to_cp)
        if dist > 1e-6:
            dir_to_cp = Vec(to_cp.x / dist, to_cp.y / dist)
        else:
            dir_to_cp = Vec(0.0, 0.0)
        speed_toward = dot(best.vel, dir_to_cp) / 1000.0
        target_angle = angle_to(cp, best.pos)
        angle_err = abs(angle_diff(target_angle, best.angle))
        return speed_toward, angle_err

    def _decode_actions(self, actions):
        actions = np.asarray(actions, dtype=np.float32).reshape(2, 4)
        actions = np.clip(actions, -1.0, 1.0)
        cmds = []
        for i in range(2):
            ax, ay, thrust_raw, special = actions[i]
            tx = (ax + 1.0) * 0.5 * MAP_W
            ty = (ay + 1.0) * 0.5 * MAP_H
            thrust = (thrust_raw + 1.0) * 50.0
            boost = special > 0.5
            shield = special < -0.5
            cmds.append(Command(Vec(tx, ty), thrust=thrust, boost=boost, shield=shield))
        return cmds

    def step(self, actions0, actions1):
        cmds0 = self._decode_actions(actions0)
        cmds1 = self._decode_actions(actions1)
        done, winner = self.game.step(cmds0, cmds1)

        crossed = [0, 0]
        idx = 0
        for team_idx in range(2):
            for pod in self.game.teams[team_idx]:
                if self.last_next_ids[idx] != pod.next_id:
                    crossed[team_idx] += 1
                self.last_next_ids[idx] = pod.next_id
                idx += 1

        score0 = self._score(0)
        score1 = self._score(1)
        delta0 = score0 - self.last_score[0]
        delta1 = score1 - self.last_score[1]
        speed0, angle0 = self._team_motion_metrics(0)
        speed1, angle1 = self._team_motion_metrics(1)

        reward0 = (delta0 - delta1) * REWARD_PROGRESS_SCALE
        reward0 += (crossed[0] - crossed[1]) * REWARD_CHECKPOINT_BONUS
        reward0 += (speed0 - speed1) * REWARD_SPEED_SCALE
        reward0 -= (angle0 - angle1) * REWARD_ANGLE_SCALE

        if done:
            if winner == 0:
                reward0 += 50.0
            else:
                reward0 -= 50.0

        self.last_score[0] = score0
        self.last_score[1] = score1

        obs0 = self._encode(0)
        obs1 = self._encode(1)
        reward1 = -reward0
        info = {"winner": winner}
        return obs0, obs1, reward0, reward1, done, info


class VectorEnv:
    def __init__(self, num_envs, **env_kwargs):
        self.envs = [MadPodEnv(**env_kwargs) for _ in range(num_envs)]
        self.num_envs = num_envs

    def set_track_params(self, **kwargs):
        for env in self.envs:
            env.set_track_params(**kwargs)

    def reset(self):
        obs0 = []
        obs1 = []
        for env in self.envs:
            o0, o1 = env.reset()
            obs0.append(o0)
            obs1.append(o1)
        return np.stack(obs0), np.stack(obs1)

    def step(self, actions0, actions1):
        next_obs0 = []
        next_obs1 = []
        rewards0 = []
        rewards1 = []
        dones = []
        infos = []
        for i, env in enumerate(self.envs):
            o0, o1, r0, r1, done, info = env.step(actions0[i], actions1[i])
            if done:
                o0, o1 = env.reset()
                info["reset"] = True
            next_obs0.append(o0)
            next_obs1.append(o1)
            rewards0.append(r0)
            rewards1.append(r1)
            dones.append(done)
            infos.append(info)
        return (
            np.stack(next_obs0),
            np.stack(next_obs1),
            np.asarray(rewards0, dtype=np.float32),
            np.asarray(rewards1, dtype=np.float32),
            np.asarray(dones, dtype=np.bool_),
            infos,
        )
