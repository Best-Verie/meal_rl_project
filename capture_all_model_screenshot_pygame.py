import argparse
import os
from pathlib import Path

import pygame

from play_best_model_pygame import BestModelPlayer


DEFAULT_SCENARIOS = [
    "baby_meal",
    "adult_hypertension",
    "adult_weight_gain",
    "adult_weight_loss",
]

DEFAULT_ALGORITHMS = [
    "ppo",
    "dqn",
    "reinforce",
]

DEFAULT_MODEL_PATHS = {
    "ppo": "models/ppo/kitchen_ppo_model.zip",
    "dqn": "models/dqn/kitchen_dqn_model.zip",
    "reinforce": "models/reinforce/kitchen_reinforce_model.pt",
}


def sanitize(text: str) -> str:
    return text.replace("/", "_").replace("\\", "_").replace(" ", "_")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_surface(surface: pygame.Surface, path: Path) -> None:
    pygame.image.save(surface, str(path))
    print(f"Saved: {path}")


def capture_episode(player: BestModelPlayer, output_dir: Path, capture_every_step: bool) -> None:
    player.draw()
    pygame.event.pump()

    if capture_every_step:
        step_file = output_dir / f"step_{player.env.current_step:02d}.png"
        save_surface(player.screen, step_file)

    while not (player.done or player.truncated):
        player.step_model()
        player.draw()
        pygame.event.pump()

        if capture_every_step:
            step_file = output_dir / f"step_{player.env.current_step:02d}.png"
            save_surface(player.screen, step_file)

    final_file = output_dir / "final.png"
    save_surface(player.screen, final_file)


def capture_case(algorithm: str, scenario: str, model_path: str, root_output_dir: Path, capture_every_step: bool) -> None:
    print(f"\n=== Capturing {algorithm.upper()} | {scenario} ===")
    case_dir = root_output_dir / sanitize(algorithm) / sanitize(scenario)
    ensure_dir(case_dir)

    player = BestModelPlayer(
        algorithm=algorithm,
        scenario_name=scenario,
        model_path=model_path,
    )

    try:
        capture_episode(player, case_dir, capture_every_step)
    finally:
        pygame.quit()


def parse_args():
    parser = argparse.ArgumentParser(description="Capture screenshots for all model/scenario combinations.")
    parser.add_argument(
        "--output-dir",
        default="screenshots",
        help="Directory where screenshots will be saved.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=DEFAULT_ALGORITHMS,
        choices=DEFAULT_ALGORITHMS,
        help="Algorithms to run.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=DEFAULT_SCENARIOS,
        help="Scenario names to run.",
    )
    parser.add_argument(
        "--capture-every-step",
        action="store_true",
        help="Save a screenshot after each step, not just the final screen.",
    )
    parser.add_argument("--ppo-model", default=DEFAULT_MODEL_PATHS["ppo"])
    parser.add_argument("--dqn-model", default=DEFAULT_MODEL_PATHS["dqn"])
    parser.add_argument("--reinforce-model", default=DEFAULT_MODEL_PATHS["reinforce"])
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    model_paths = {
        "ppo": args.ppo_model,
        "dqn": args.dqn_model,
        "reinforce": args.reinforce_model,
    }

    for algorithm in args.algorithms:
        model_path = model_paths[algorithm]
        if not Path(model_path).exists():
            print(f"Skipping {algorithm.upper()} because model file was not found: {model_path}")
            continue

        for scenario in args.scenarios:
            capture_case(
                algorithm=algorithm,
                scenario=scenario,
                model_path=model_path,
                root_output_dir=output_dir,
                capture_every_step=args.capture_every_step,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
