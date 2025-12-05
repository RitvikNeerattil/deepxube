from deepxube.base.env import NumberLinkDeepXubeEnv
from deepxube.tests.test_env import test_env
from argparse import ArgumentParser


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--num_states', type=int, required=True, help="")
    parser.add_argument('--step_max', type=int, required=True, help="")
    parser.add_argument('--width', type=int, default=7, help="")
    parser.add_argument('--height', type=int, default=7, help="")
    parser.add_argument('--num_colors', type=int, default=5, help="")
    args = parser.parse_args()

    env = NumberLinkDeepXubeEnv(width=args.width, height=args.height, num_colors=args.num_colors)
    test_env(env, args.num_states, args.step_max)


if __name__ == "__main__":
    main()
