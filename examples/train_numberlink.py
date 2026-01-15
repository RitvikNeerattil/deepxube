from argparse import ArgumentParser
import torch

from deepxube.training.train_utils import TrainArgs
from deepxube.training.train_heur import train
from deepxube.base.updater import UpArgs, UpdateHeur, UpHeurArgs
from deepxube.updater.updaters import UpdateHeurBWASEnum, UpBWASArgs
from deepxube.implementations.numberlink import NumberLinkDeepXubeEnv, NumberLinkNNetParV


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--heur_type', type=str, default="V", help="Heuristic type (V for value function)")
    parser.add_argument('--nnet_dir', type=str, required=True, help="Directory to save the neural network model")
    
    # Environment args
    parser.add_argument('--width', type=int, default=7, help="Width of the NumberLink grid")
    parser.add_argument('--height', type=int, default=7, help="Height of the NumberLink grid")
    parser.add_argument('--num_colors', type=int, default=5, help="Number of colors in the NumberLink puzzle")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")

    # Train args
    parser.add_argument('--batch_size', type=int, default=100, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--lr_d', type=float, default=0.99999, help="Learning rate decay")
    parser.add_argument('--max_itrs', type=int, default=1000000, help="Maximum training iterations")
    parser.add_argument('--balance', action='store_true', default=False, help="Balance training data")
    parser.add_argument('--targ_up_searches', type=int, default=0, help="Target update searches")
    parser.add_argument('--display', type=int, default=100, help="Display frequency")

    # Updater args
    parser.add_argument('--step_max', type=int, required=True, help="Maximum number of steps for random walks")
    parser.add_argument('--up_itrs', type=int, default=100, help="Updater iterations")
    parser.add_argument('--up_gen_itrs', type=int, default=100, help="Updater generation iterations")
    parser.add_argument('--up_procs', type=int, default=1, help="Number of processes for the updater")
    parser.add_argument('--up_search_itrs', type=int, default=100, help="Updater search iterations")
    parser.add_argument('--up_batch_size', type=int, default=100, help="Updater batch size")
    parser.add_argument('--up_nnet_batch_size', type=int, default=10000, help="Updater nnet batch size")
    parser.add_argument('--verbose', action='store_true', default=True, help="Enable verbose logging for the updater.")
    parser.add_argument('--sync_main', action='store_true', default=False, help="If true, number of processes can affect order in which data is seen")

    # Other args
    parser.add_argument('--backup', type=int, default=1, help="Backup")
    parser.add_argument('--eps', type=float, default=0.2, help="Epsilon for epsilon-greedy")
    parser.add_argument('--weight', type=float, default=1.0, help="Search weight for BWAS")
    parser.add_argument('--rb', type=int, default=1, help="Replay buffer size")
    parser.add_argument('--debug', action='store_true', default=False, help="Enable debug mode")
    
    args = parser.parse_args()

    up_args: UpArgs = UpArgs(args.step_max, args.up_itrs, args.up_gen_itrs, args.up_procs, args.up_search_itrs,
                             args.up_batch_size, args.up_nnet_batch_size, args.sync_main, up_v=args.verbose)
    
    env = NumberLinkDeepXubeEnv(width=args.width, height=args.height, num_colors=args.num_colors)
    
    updater: UpdateHeur
    if args.heur_type.upper() == "V":
        nnet_par = NumberLinkNNetParV(width=args.width, height=args.height, num_colors=args.num_colors, device=args.device)
        up_heur_args = UpHeurArgs(up_args, ub_heur_solns=False, backup=args.backup)
        up_bwas_args = UpBWASArgs(up_heur_args, weight=args.weight, eps=args.eps)
        updater = UpdateHeurBWASEnum(env, up_bwas_args, nnet_par)
    else:
        raise ValueError(f"Unknown or unsupported heuristic type '{args.heur_type}'. Only 'V' is supported for now.")

    train_args: TrainArgs = TrainArgs(args.batch_size, args.lr, args.lr_d, args.max_itrs, args.balance, args.rb,
                                      args.targ_up_searches, args.display)
    
    print("Starting training for NumberLink...")
    train(updater, args.nnet_dir, train_args, debug=args.debug)
    print("Training finished.")


if __name__ == "__main__":
    main()
