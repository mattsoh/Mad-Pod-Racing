import math
import numpy as np
from sim.physics import (
    Vec,
    MAP_W,
    MAP_H,
    FRICTION,
    CHECKPOINT_RADIUS,
    POD_RADIUS,
    norm,
    normalize,
    angle_to,
    angle_diff,
)


class BaselineBot:
    def __init__(self, checkpoints):
        self.reset(checkpoints)

    def reset(self, checkpoints):
        self.checkpoints = checkpoints
        self.longest_stretch = self._compute_longest_stretch() * 0.70
        self.boost_wait = [30, 30]
        self.boost_on_first = [False, True]
        self.shield_cd = [0, 0]
        self.turn = 0

    def _compute_longest_stretch(self):
        if not self.checkpoints:
            return 0.0
        longest = 0.0
        n = len(self.checkpoints)
        for i in range(n):
            d = norm(self.checkpoints[i] - self.checkpoints[(i + 1) % n])
            longest = max(longest, d)
        return longest

    @staticmethod
    def _closest_point_to_line(a, b, p):
        a_to_p = p - a
        a_to_b = b - a
        atb2 = norm(a_to_b)
        atb2 = atb2 * atb2
        if atb2 < 1e-6:
            return p
        t = (a_to_p.x * a_to_b.x + a_to_p.y * a_to_b.y) / atb2
        return Vec(a.x + a_to_b.x * t, a.y + a_to_b.y * t)

    @staticmethod
    def _abs_angle_diff(a, b):
        return abs(angle_diff(a, b))

    def _is_entering_checkpoint_soon(self, pod):
        cp = self.checkpoints[pod.next_id]
        vel = Vec(pod.vel.x, pod.vel.y)
        pos = Vec(pod.pos.x, pod.pos.y)
        for _ in range(6):
            vel = vel * FRICTION
            pos = pos + vel
            if norm(cp - pos) <= CHECKPOINT_RADIUS:
                return True
        return False

    def _thrust_from_angle(self, abs_degrees):
        thrust = 10
        if abs_degrees < 90.0:
            thrust = int(20 + math.ceil(80.0 * math.cos(abs_degrees / 100.0)))
        if thrust < 0:
            thrust = 0
        if thrust > 100:
            thrust = 100
        return thrust

    def _should_shield(self, pod, enemies, pod_idx):
        if self.shield_cd[pod_idx] > 0:
            return False
        next_pos = pod.pos + pod.vel * FRICTION
        for enemy in enemies:
            rel = pod.vel - enemy.vel
            rel_speed = norm(rel)
            if rel_speed < 220.0:
                continue
            enemy_next = enemy.pos + enemy.vel * FRICTION
            if norm(next_pos - enemy_next) <= POD_RADIUS * 2.0:
                return True
        return False

    def _encode_action(self, target, thrust, boost, shield):
        tx = min(max(target.x, 0.0), MAP_W)
        ty = min(max(target.y, 0.0), MAP_H)
        ax = tx / MAP_W * 2.0 - 1.0
        ay = ty / MAP_H * 2.0 - 1.0
        thrust_raw = thrust / 50.0 - 1.0
        special = 0.0
        if boost:
            special = 1.0
        elif shield:
            special = -1.0
        return [
            float(np.clip(ax, -1.0, 1.0)),
            float(np.clip(ay, -1.0, 1.0)),
            float(np.clip(thrust_raw, -1.0, 1.0)),
            float(np.clip(special, -1.0, 1.0)),
        ]

    def act(self, game, team_idx):
        pods = game.teams[team_idx]
        enemies = game.teams[1 - team_idx]
        actions = []

        for i, pod in enumerate(pods):
            if self.shield_cd[i] > 0:
                self.shield_cd[i] -= 1
            if self.boost_wait[i] > 0:
                self.boost_wait[i] -= 1

            cp = self.checkpoints[pod.next_id]
            next_cp = self.checkpoints[(pod.next_id + 1) % len(self.checkpoints)]
            current_pos = pod.pos
            current_dist = norm(cp - current_pos)

            abs_degrees = self._abs_angle_diff(angle_to(cp, current_pos), pod.angle)
            target = cp

            future_pos = current_pos + pod.vel * FRICTION
            distance_travelled = norm(pod.vel * FRICTION)

            if self._is_entering_checkpoint_soon(pod):
                target = next_cp
                abs_degrees = self._abs_angle_diff(angle_to(next_cp, current_pos), pod.angle)
            elif distance_travelled > 50.0 and abs_degrees < 70.0:
                if norm(cp - future_pos) < current_dist:
                    aux = self._closest_point_to_line(current_pos, cp, future_pos)
                    target = aux + (aux - future_pos)

            thrust = self._thrust_from_angle(abs_degrees)

            use_boost = False
            if game.boost_available[team_idx]:
                if self.boost_on_first[i] and self.turn == 0:
                    use_boost = True
                elif self.boost_wait[i] <= 0 and abs_degrees < 5.0 and current_dist > self.longest_stretch:
                    use_boost = True

            use_shield = False
            if not use_boost and self._should_shield(pod, enemies, i):
                use_shield = True
                self.shield_cd[i] = 3

            actions.append(self._encode_action(target, thrust, use_boost, use_shield))

        self.turn += 1
        return np.asarray(actions, dtype=np.float32).reshape(-1)
