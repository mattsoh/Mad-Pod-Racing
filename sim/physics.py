import math
from dataclasses import dataclass

MAP_W = 16000.0
MAP_H = 9000.0
CHECKPOINT_RADIUS = 600.0
POD_RADIUS = 400.0
MAX_ROTATION_DEG = 18.0
FRICTION = 0.85
MAX_THRUST = 100.0
BOOST_THRUST = 650.0
MIN_IMPULSE = 120.0


@dataclass
class Vec:
    x: float
    y: float

    def __add__(self, other):
        return Vec(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec(self.x - other.x, self.y - other.y)

    def __mul__(self, k: float):
        return Vec(self.x * k, self.y * k)


def dot(a: Vec, b: Vec) -> float:
    return a.x * b.x + a.y * b.y


def norm2(a: Vec) -> float:
    return dot(a, a)


def norm(a: Vec) -> float:
    return math.sqrt(norm2(a))


def normalize(a: Vec) -> Vec:
    n = norm(a)
    if n < 1e-9:
        return Vec(0.0, 0.0)
    return Vec(a.x / n, a.y / n)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def normalize_angle(angle: float) -> float:
    angle = angle % 360.0
    if angle < 0.0:
        angle += 360.0
    return angle


def angle_diff(target: float, current: float) -> float:
    d = (target - current + 540.0) % 360.0 - 180.0
    return d


def angle_to(target: Vec, source: Vec) -> float:
    return normalize_angle(math.degrees(math.atan2(target.y - source.y, target.x - source.x)))


def angle_to_vec(angle_deg: float) -> Vec:
    rad = math.radians(angle_deg)
    return Vec(math.cos(rad), math.sin(rad))


@dataclass
class Pod:
    pos: Vec
    vel: Vec
    angle: float
    next_id: int
    lap: int = 0
    shield_cd: int = 0

    @property
    def mass(self) -> float:
        return 10.0 if self.shield_cd > 0 else 1.0

    def rotate_towards(self, target: Vec, max_deg: float) -> None:
        target_angle = angle_to(target, self.pos)
        diff = angle_diff(target_angle, self.angle)
        diff = clamp(diff, -max_deg, max_deg)
        self.angle = normalize_angle(self.angle + diff)

    def apply_thrust(self, thrust: float) -> None:
        direction = angle_to_vec(self.angle)
        self.vel = self.vel + direction * thrust

    def move(self, dt: float) -> None:
        self.pos = self.pos + self.vel * dt

    def apply_friction_and_round(self) -> None:
        self.vel = Vec(self.vel.x * FRICTION, self.vel.y * FRICTION)
        self.vel = Vec(
            float(math.trunc(float(self.vel.x))),
            float(math.trunc(float(self.vel.y))),
        )
        self.pos = Vec(float(int(round(self.pos.x))), float(int(round(self.pos.y))))

    def advance_checkpoint(self, checkpoints) -> bool:
        cp = checkpoints[self.next_id]
        if norm(cp - self.pos) <= CHECKPOINT_RADIUS:
            self.next_id = (self.next_id + 1) % len(checkpoints)
            if self.next_id == 0:
                self.lap += 1
            return True
        return False


def time_to_collision(p1: Pod, p2: Pod, remaining: float):
    dp = p2.pos - p1.pos
    dv = p2.vel - p1.vel
    r = POD_RADIUS * 2.0
    a = dot(dv, dv)
    if a < 1e-9:
        return None
    b = 2.0 * dot(dp, dv)
    c = dot(dp, dp) - r * r
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    t = (-b - math.sqrt(disc)) / (2.0 * a)
    if 0.0 <= t <= remaining:
        return t
    return None


def resolve_collision(p1: Pod, p2: Pod) -> None:
    n = normalize(p2.pos - p1.pos)
    rel = p2.vel - p1.vel
    vrel = dot(rel, n)
    if vrel >= 0.0:
        return
    impulse = -(1.0 + 1.0) * vrel / (1.0 / p1.mass + 1.0 / p2.mass)
    if impulse < MIN_IMPULSE:
        impulse = MIN_IMPULSE
    p1.vel = p1.vel - n * (impulse / p1.mass)
    p2.vel = p2.vel + n * (impulse / p2.mass)


def resolve_collisions(pods, dt: float = 1.0) -> None:
    time = 0.0
    max_iters = 10
    while time < dt and max_iters > 0:
        max_iters -= 1
        soonest_t = None
        pair = None
        remaining = dt - time
        for i in range(len(pods)):
            for j in range(i + 1, len(pods)):
                t = time_to_collision(pods[i], pods[j], remaining)
                if t is not None and (soonest_t is None or t < soonest_t):
                    soonest_t = t
                    pair = (i, j)
        if soonest_t is None:
            for pod in pods:
                pod.move(remaining)
            break
        for pod in pods:
            pod.move(soonest_t)
        resolve_collision(pods[pair[0]], pods[pair[1]])
        time += soonest_t
