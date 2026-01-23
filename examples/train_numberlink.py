import os
import sys
import types
from argparse import ArgumentParser

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_CPP_MIN_VLOG_LEVEL", "0")
os.environ.setdefault("GLOG_minloglevel", "3")

import torch


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
    parser.add_argument("--device", type=str, default="auto")

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
    parser.add_argument("--up_gen_itrs", type=int, default=10)      # IMPORTANT
    parser.add_argument("--up_procs", type=int, default=2)
    parser.add_argument("--up_search_itrs", type=int, default=20)   # IMPORTANT
    parser.add_argument("--up_batch_size", type=int, default=32)    # IMPORTANT
    parser.add_argument("--up_nnet_batch_size", type=int, default=4096)
    parser.add_argument("--sync_main", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=True)

    # -------- misc --------
    parser.add_argument("--backup", type=int, default=1)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--rb", type=int, default=1)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--no_tensorboard", action="store_true", default=False)
    parser.add_argument("--smoke_test", action="store_true", default=False)
    parser.add_argument("--no_shm", action="store_true", default=False)

    args = parser.parse_args()

    if args.no_tensorboard:
        os.environ.setdefault("TENSORBOARD_NO_TENSORFLOW", "1")

        import importlib.machinery

        tf_mod = types.ModuleType("tensorflow")
        tf_mod.__spec__ = importlib.machinery.ModuleSpec("tensorflow", loader=None)
        sys.modules["tensorflow"] = tf_mod

        tb_pkg = types.ModuleType("tensorboard")
        tb_pkg.__spec__ = importlib.machinery.ModuleSpec("tensorboard", loader=None)
        sys.modules["tensorboard"] = tb_pkg

        tb_mod = types.ModuleType("torch.utils.tensorboard")

        class SummaryWriter:
            def __init__(self, *a, **k):
                pass

            def add_scalar(self, *a, **k):
                pass

            def flush(self):
                pass

            def close(self):
                pass

        tb_mod.SummaryWriter = SummaryWriter
        sys.modules["torch.utils.tensorboard"] = tb_mod

    if args.no_shm:
        os.environ["DEEPXUBE_NO_SHM"] = "1"

    from deepxube.training.train_utils import TrainArgs
    from deepxube.training.train_heur import train
    from deepxube.base.updater import UpArgs, UpdateHeur, UpHeurArgs
    from deepxube.updater.updaters import UpdateHeurBWASEnum, UpBWASArgs

    from deepxube.implementations.numberlink import (
        NumberLinkDeepXubeEnv,
        NumberLinkNNetParV,
    )
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        args.device = "cpu"

    # ===== ENV =====
    env = NumberLinkDeepXubeEnv(
        width=args.width,
        height=args.height,
        num_colors=args.num_colors,
    )
    if args.smoke_test:
        nnet_par_smoke = NumberLinkNNetParV(
            width=args.width,
            height=args.height,
            num_colors=args.num_colors,
            device=args.device,
        )
        states = env.get_start_states(1)
        goals = env.sample_goal(states, states)
        with torch.no_grad():
            inputs = nnet_par_smoke.to_torch(states, goals)
            nnet = nnet_par_smoke.get_nnet()
            nnet.eval()
            out = nnet(inputs)
        print(f"Smoke test OK. Output shape: {tuple(out.shape)}")
        return

    # ===== UPDATER =====
    num_gen = args.batch_size * args.up_gen_itrs
    if args.up_search_itrs <= 0:
        raise ValueError("--up_search_itrs must be > 0")
    if num_gen % args.up_search_itrs != 0:
        adjusted = min(args.up_search_itrs, num_gen)
        while adjusted > 1 and (num_gen % adjusted) != 0:
            adjusted -= 1
        print(
            "Adjusting --up_search_itrs from"
            f" {args.up_search_itrs} to {adjusted} so num_gen ({num_gen}) is divisible."
        )
        args.up_search_itrs = adjusted

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
