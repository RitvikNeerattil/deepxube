import os
from argparse import ArgumentParser

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_CPP_MIN_VLOG_LEVEL", "0")
os.environ.setdefault("GLOG_minloglevel", "3")

import torch

from deepxube.training.train_utils import TrainArgs
from deepxube.training.train_heur import train
from deepxube.base.updater import UpArgs, UpdateHeur, UpHeurArgs
from deepxube.updater.updaters import UpdateHeurBWASEnum, UpBWASArgs
from deepxube.implementations.numberlink import (
    NumberLinkDeepXubeEnv,
    NumberLinkNNetParV,
)


def main():
    parser = ArgumentParser()

    # -------- basic --------
    parser.add_argument("--heur_type", type=str, default="V")
    parser.add_argument("--nnet_dir", type=str, required=True)

    # -------- env --------
    parser.add_argument("--width", type=int, default=7)
    parser.add_argument("--height", type=int, default=7)
    parser.add_argument("--num_colors", type=int, default=5)

    # -------- device --------
    parser.add_argument("--device", type=str, default="cuda")

    # -------- training --------
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_d", type=float, default=0.99999)
    parser.add_argument("--max_itrs", type=int, default=100000)
    parser.add_argument("--balance", action="store_true", default=False)
    parser.add_argument("--targ_up_searches", type=int, default=0)
    parser.add_argument("--display", type=int, default=10)

    # -------- BWAS / updater --------
    parser.add_argument("--step_max", type=int, required=True)
    parser.add_argument("--up_itrs", type=int, default=50)
    parser.add_argument("--up_gen_itrs", type=int, default=10)      # ↓ IMPORTANT
    parser.add_argument("--up_procs", type=int, default=2)
    parser.add_argument("--up_search_itrs", type=int, default=20)   # ↓ IMPORTANT
    parser.add_argument("--up_batch_size", type=int, default=32)    # ↓ IMPORTANT
    parser.add_argument("--up_nnet_batch_size", type=int, default=4096)
    parser.add_argument("--sync_main", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=True)

    # -------- misc --------
    parser.add_argument("--backup", type=int, default=1)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--rb", type=int, default=1)
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()

    # ===== HARD CUDA ENFORCEMENT =====
    if args.device == "cuda":
        assert torch.cuda.is_available(), (
            "CUDA requested but torch.cuda.is_available() == False"
        )
        torch.cuda.set_device(0)

    os.makedirs(args.nnet_dir, exist_ok=True)

    # ===== ENV =====
    env = NumberLinkDeepXubeEnv(
        width=args.width,
        height=args.height,
        num_colors=args.num_colors,
    )

    # ===== UPDATER =====
    up_args = UpArgs(
        args.step_max,
        args.up_itrs,
        args.up_gen_itrs,
        args.up_procs,
        args.up_search_itrs,
        args.up_batch_size,
        args.up_nnet_batch_size,
        args.sync_main,
        up_v=args.verbose,
    )

    if args.heur_type.upper() != "V":
        raise ValueError("Only V heuristic is supported")

    nnet_par = NumberLinkNNetParV(
        width=args.width,
        height=args.height,
        num_colors=args.num_colors,
        device=args.device,
    )

    up_heur_args = UpHeurArgs(up_args, ub_heur_solns=False, backup=args.backup)
    up_bwas_args = UpBWASArgs(up_heur_args, weight=args.weight, eps=args.eps)
    updater: UpdateHeur = UpdateHeurBWASEnum(env, up_bwas_args, nnet_par)

    train_args = TrainArgs(
        args.batch_size,
        args.lr,
        args.lr_d,
        args.max_itrs,
        args.balance,
        args.rb,
        args.targ_up_searches,
        args.display,
    )

    print("Starting training for NumberLink...")
    train(updater, args.nnet_dir, train_args, debug=args.debug)
    print("Training finished.")


if __name__ == "__main__":
    main()
