import argparse
import json
import math
import pygame
from sim.physics import MAP_W, MAP_H, POD_RADIUS, CHECKPOINT_RADIUS


def load_replay(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def world_to_screen(x, y, scale, offset):
    return int(x * scale + offset[0]), int(y * scale + offset[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--speed", type=int, default=1)
    args = parser.parse_args()

    data = load_replay(args.replay)
    checkpoints = data["map"]["checkpoints"]
    frames = data["frames"]

    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
    pygame.display.set_caption("Mad Pod Racing Replay")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    scale = min(screen.get_width() / MAP_W, screen.get_height() / MAP_H)
    offset = (
        (screen.get_width() - MAP_W * scale) * 0.5,
        (screen.get_height() - MAP_H * scale) * 0.5,
    )

    idx = 0
    paused = False
    speed = max(1, args.speed)
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT:
                    idx = min(idx + 1, len(frames) - 1)
                elif event.key == pygame.K_LEFT:
                    idx = max(idx - 1, 0)
                elif event.key == pygame.K_UP:
                    speed = min(8, speed + 1)
                elif event.key == pygame.K_DOWN:
                    speed = max(1, speed - 1)

        if not paused:
            idx = min(idx + speed, len(frames) - 1)

        screen.fill((12, 16, 24))

        for cp in checkpoints:
            x, y = world_to_screen(cp[0], cp[1], scale, offset)
            pygame.draw.circle(screen, (60, 70, 80), (x, y), int(CHECKPOINT_RADIUS * scale), 2)

        pods = frames[idx]["pods"]
        for i, pod in enumerate(pods):
            x, y, angle, next_id, lap, shield_cd = pod
            color = (60, 160, 255) if i < 2 else (255, 80, 80)
            sx, sy = world_to_screen(x, y, scale, offset)
            pygame.draw.circle(screen, color, (sx, sy), int(POD_RADIUS * scale))

            rad = math.radians(angle)
            hx = sx + math.cos(rad) * POD_RADIUS * scale
            hy = sy + math.sin(rad) * POD_RADIUS * scale
            pygame.draw.line(screen, (240, 240, 240), (sx, sy), (hx, hy), 2)

        text = font.render(f"Frame {idx + 1}/{len(frames)}  Speed {speed}x", True, (230, 230, 230))
        legend = font.render("Blue = you, Red = opponent", True, (200, 200, 200))
        screen.blit(text, (10, 10))
        screen.blit(legend, (10, 34))

        pygame.display.flip()
        clock.tick(max(10, args.fps))

    pygame.quit()


if __name__ == "__main__":
    main()
