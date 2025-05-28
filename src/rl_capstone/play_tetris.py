import argparse
import warnings
import os
import sys

# Suppress the pygame pkg_resources deprecation warning
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")  # noqa: E402

if __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from rl_capstone.tetris_environment import Tetris_Env  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Play the Tetris game")
    parser.add_argument(
        "--render-mode",
        default="human",
        choices=["human", "rgb_array"],
        help="Rendering mode for the environment",
    )
    args = parser.parse_args()
    env = Tetris_Env(render_mode=args.render_mode)
    env.play_game()


if __name__ == "__main__":
    main()
