import json
from .physics import MAP_W, MAP_H


class ReplayRecorder:
    def __init__(self, checkpoints, laps):
        self.checkpoints = checkpoints
        self.laps = laps
        self.frames = []

    def add_frame(self, game):
        pods = []
        for team in game.teams:
            for pod in team:
                pods.append(
                    [
                        pod.pos.x,
                        pod.pos.y,
                        pod.angle,
                        pod.next_id,
                        pod.lap,
                        pod.shield_cd,
                    ]
                )
        self.frames.append({"pods": pods, "boost": list(game.boost_available)})

    def to_json(self, path):
        data = {
            "map": {
                "width": MAP_W,
                "height": MAP_H,
                "checkpoints": [[cp.x, cp.y] for cp in self.checkpoints],
                "laps": self.laps,
            },
            "frames": self.frames,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
