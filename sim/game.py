from dataclasses import dataclass
import random
from .physics import (
    Vec,
    Pod,
    normalize,
    angle_to,
    MAX_ROTATION_DEG,
    BOOST_THRUST,
    MAX_THRUST,
    clamp,
    resolve_collisions,
)


@dataclass
class Command:
    target: Vec
    thrust: float = 0.0
    boost: bool = False
    shield: bool = False


class GameState:
    def __init__(self, laps, checkpoints, seed=None):
        self.rng = random.Random(seed)
        self.reset(checkpoints, laps)

    def reset(self, checkpoints, laps):
        self.laps = laps
        self.checkpoints = checkpoints
        self.checkpoint_count = len(checkpoints)
        self.turn_index = 0
        self.boost_available = [True, True]
        self.no_progress = [0, 0]
        self.teams = [[], []]
        self._spawn_pods()

    def _spawn_pods(self):
        cp0 = self.checkpoints[0]
        cp1 = self.checkpoints[1 % self.checkpoint_count]
        direction = normalize(cp1 - cp0)
        if direction.x == 0.0 and direction.y == 0.0:
            direction = Vec(1.0, 0.0)
        perp = Vec(-direction.y, direction.x)
        base_angle = angle_to(cp1, cp0)
        offsets = [600.0, -600.0]
        enemy_offsets = [1200.0, -1200.0]

        team0 = []
        team1 = []
        for i in range(2):
            pos0 = cp0 + perp * offsets[i]
            pos1 = cp0 + perp * enemy_offsets[i]
            team0.append(Pod(pos0, Vec(0.0, 0.0), base_angle, 1))
            team1.append(Pod(pos1, Vec(0.0, 0.0), base_angle, 1))
        self.teams = [team0, team1]

    def _apply_commands(self, team_idx, cmds):
        max_rot = 360.0 if self.turn_index == 0 else MAX_ROTATION_DEG
        for i in range(2):
            pod = self.teams[team_idx][i]
            cmd = cmds[i]
            pod.rotate_towards(cmd.target, max_rot)

            thrust = 0.0
            if cmd.shield:
                pod.shield_cd = 3
                thrust = 0.0
            elif pod.shield_cd > 0:
                thrust = 0.0
            else:
                if cmd.boost and self.boost_available[team_idx]:
                    thrust = BOOST_THRUST
                    self.boost_available[team_idx] = False
                elif cmd.boost:
                    thrust = MAX_THRUST
                else:
                    thrust = clamp(cmd.thrust, 0.0, MAX_THRUST)
            pod.apply_thrust(thrust)

    def step(self, cmds0, cmds1):
        self._apply_commands(0, cmds0)
        self._apply_commands(1, cmds1)

        all_pods = self.teams[0] + self.teams[1]
        resolve_collisions(all_pods, 1.0)

        for pod in all_pods:
            pod.apply_friction_and_round()

        advanced = [False, False]
        winner = None
        for team_idx in range(2):
            for pod in self.teams[team_idx]:
                if pod.advance_checkpoint(self.checkpoints):
                    advanced[team_idx] = True
                    if pod.lap >= self.laps and winner is None:
                        winner = team_idx

        for team_idx in range(2):
            if advanced[team_idx]:
                self.no_progress[team_idx] = 0
            else:
                self.no_progress[team_idx] += 1
            if self.no_progress[team_idx] >= 100 and winner is None:
                winner = 1 - team_idx

        for pod in all_pods:
            if pod.shield_cd > 0:
                pod.shield_cd -= 1

        self.turn_index += 1
        done = winner is not None
        return done, winner
