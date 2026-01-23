from typing import List, Optional
import os
import random
import numpy as np
from numpy.typing import NDArray

import torch
from torch import Tensor

from deepxube.base.env import Action, EnvEnumerableActs, EnvStartGoalRW, Goal, State
from deepxube.base.heuristic import HeurNNetModule, HeurNNetV
from deepxube.nnet.pytorch_models import FullyConnectedModel

from numberlink.config import GeneratorConfig, VariantConfig
from numberlink.vector_env import NumberLinkRGBVectorEnv


# Lightweight, picklable wrapper to bypass shared_memory in small runs.
class LocalNDArray:
    def __init__(self, array: NDArray):
        self.array = array

    def close(self) -> None:
        pass

    def unlink(self) -> None:
        pass


def np_to_local_nd(arr: NDArray) -> LocalNDArray:
    return LocalNDArray(arr)


def _patch_no_shm() -> None:
    if os.environ.get("DEEPXUBE_NO_SHM", "0") != "1":
        return
    try:
        from deepxube.base import updater as updater_mod
        from deepxube.nnet import nnet_utils as nnet_utils_mod
        from deepxube.utils import data_utils as data_utils_mod
        updater_mod.np_to_shnd = np_to_local_nd
        nnet_utils_mod.np_to_shnd = np_to_local_nd
        data_utils_mod.np_to_shnd = np_to_local_nd
    except Exception:
        pass


_patch_no_shm()

#state
class NumberLinkState(State):
    __slots__ = ("grid_codes", "lane_v", "lane_h", "closed", "steps", "_hash")

    def __init__(
        self,
        grid_codes: NDArray[np.uint8],
        lane_v: NDArray[np.uint8],
        lane_h: NDArray[np.uint8],
        closed: NDArray[np.bool_],
        steps: int,
    ):
        self.grid_codes = grid_codes
        self.lane_v = lane_v
        self.lane_h = lane_h
        self.closed = closed
        self.steps = steps
        self._hash: Optional[int] = None

    def __hash__(self):
        if self._hash is None:
            self._hash = hash((
                self.grid_codes.tobytes(),
                self.lane_v.tobytes(),
                self.lane_h.tobytes(),
                self.closed.tobytes(),
            ))
        return self._hash

    def __eq__(self, other):
        return (
            isinstance(other, NumberLinkState)
            and np.array_equal(self.grid_codes, other.grid_codes)
            and np.array_equal(self.lane_v, other.lane_v)
            and np.array_equal(self.lane_h, other.lane_h)
            and np.array_equal(self.closed, other.closed)
        )


#action
class NumberLinkAction(Action):
    __slots__ = ("action",)

    def __init__(self, action: int):
        self.action = int(action)

    def __hash__(self):
        return self.action

    def __eq__(self, other):
        return isinstance(other, NumberLinkAction) and self.action == other.action


class NumberLinkGoal(Goal):
    pass


#env
class NumberLinkDeepXubeEnv(
    EnvStartGoalRW[NumberLinkState, NumberLinkAction, NumberLinkGoal],
    EnvEnumerableActs,
):
    def __init__(self, width=7, height=7, num_colors=5):
        super().__init__()
        self.width = width
        self.height = height
        self.num_colors = num_colors

        tmp = NumberLinkRGBVectorEnv(
            num_envs=1,
            generator=GeneratorConfig(
                width=width,
                height=height,
                colors=num_colors,
            ),
            variant=VariantConfig(cell_switching_mode=True),
        )
        self.action_size = tmp.single_action_space.n

    def get_start_states(self, n):
        env = NumberLinkRGBVectorEnv(
            num_envs=n,
            generator=GeneratorConfig(
                width=self.width,
                height=self.height,
                colors=self.num_colors,
            ),
            variant=VariantConfig(cell_switching_mode=True),
        )
        env.reset()

        return [
            NumberLinkState(
                env._grid_codes[i].copy(),
                env._lane_v[i].copy(),
                env._lane_h[i].copy(),
                env._closed[i].copy(),
                int(env._step_count[i]),
            )
            for i in range(n)
        ]

    def sample_goal(self, states_start, states_goal):
        return [NumberLinkGoal() for _ in states_start]

    def get_state_actions(self, states):
        env = self._env_from_states(states)
        masks = env._compute_action_mask()
        return [
            [NumberLinkAction(a) for a in np.flatnonzero(masks[i])]
            for i in range(len(states))
        ]

    def get_state_action_rand(self, states):
        return [
            random.choice(a) if a else NumberLinkAction(0)
            for a in self.get_state_actions(states)
        ]

    def next_state(self, states, actions):
        env = self._env_from_states(states)
        act = np.array([a.action for a in actions], dtype=np.int64)
        _, rewards, _, _, _ = env.step(act)

        return (
            [
                NumberLinkState(
                    env._grid_codes[i].copy(),
                    env._lane_v[i].copy(),
                    env._lane_h[i].copy(),
                    env._closed[i].copy(),
                    int(env._step_count[i]),
                )
                for i in range(len(states))
            ],
            [-float(r) for r in rewards],
        )

    def is_solved(self, states, goals):
        env = self._env_from_states(states)
        return list(env._compute_solved_mask())

    def _env_from_states(self, states):
        env = NumberLinkRGBVectorEnv(
            num_envs=len(states),
            generator=GeneratorConfig(
                width=self.width,
                height=self.height,
                colors=self.num_colors,
            ),
            variant=VariantConfig(cell_switching_mode=True),
        )
        env.reset()
        for i, s in enumerate(states):
            env._grid_codes[i] = s.grid_codes
            env._lane_v[i] = s.lane_v
            env._lane_h[i] = s.lane_h
            env._closed[i] = s.closed
            env._step_count[i] = s.steps
        return env


class OneHot(torch.nn.Module):
    def __init__(self, data_dim: int, one_hot_depth: int) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.one_hot_depth = one_hot_depth

    def forward(self, x: Tensor) -> Tensor:
        x = torch.nn.functional.one_hot(x.long(), self.one_hot_depth)
        x = x.float()
        return x.view(-1, self.data_dim * self.one_hot_depth)


#network
class NumberLinkNNet(HeurNNetModule):
    def __init__(self, width, height, one_hot_depth, device):
        super().__init__()
        state_dim = width * height
        self.grid_proc = OneHot(state_dim, one_hot_depth)
        self.lv_proc = OneHot(state_dim, one_hot_depth)
        self.lh_proc = OneHot(state_dim, one_hot_depth)

        input_dim = (state_dim * one_hot_depth * 3) + state_dim
        self.fc = FullyConnectedModel(
            input_dim,
            [512, 256, 1],
            ["RELU", "RELU", "LINEAR"],
            batch_norms=[True, True, False],
        )
        self.to(device)

    def forward(self, states_goals_l: List[Tensor]) -> Tensor:
        grid, lv, lh, closed = states_goals_l[0:4]
        grid_proc = self.grid_proc(grid)
        lv_proc = self.lv_proc(lv)
        lh_proc = self.lh_proc(lh)
        closed_proc = closed.float().view(closed.shape[0], -1)
        x = torch.cat((grid_proc, lv_proc, lh_proc, closed_proc), dim=1)
        return self.fc(x)


class NumberLinkNNetParV(HeurNNetV[NumberLinkState, NumberLinkGoal]):
    def __init__(self, width, height, num_colors=None, device="cuda"):
        super().__init__()
        self.width = width
        self.height = height
        self.num_colors = num_colors
        self.device = torch.device("cuda" if device == "cuda" else "cpu")
        self.on_gpu = self.device.type == "cuda"  # REQUIRED

    def get_nnet(self):
        one_hot_depth = (self.num_colors or 0) + 1
        return NumberLinkNNet(self.width, self.height, one_hot_depth, self.device)

    def to_torch(self, states, goals):
        B = len(states)

        grid = torch.zeros((B, self.height, self.width), device=self.device, dtype=torch.int64)
        lv = torch.zeros_like(grid)
        lh = torch.zeros_like(grid)
        closed = torch.zeros((B, self.height, self.width), device=self.device, dtype=torch.float32)

        for i, s in enumerate(states):
            grid[i] = torch.from_numpy(s.grid_codes).to(
                self.device, dtype=torch.int64, non_blocking=True
            )
            lv[i] = torch.from_numpy(s.lane_v).to(
                self.device, dtype=torch.int64, non_blocking=True
            )
            lh[i] = torch.from_numpy(s.lane_h).to(
                self.device, dtype=torch.int64, non_blocking=True
            )

            for c, done in enumerate(s.closed):
                if done:
                    code = c + 1
                    mask = (
                        (grid[i] == code)
                        | (lv[i] == code)
                        | (lh[i] == code)
                    )
                    closed[i][mask] = 1.0

        goals_t = torch.zeros((B,), device=self.device)

        return [grid, lv, lh, closed, goals_t]

    # REQUIRED BY ABSTRACT BASE CLASS
    def to_np(self, states, goals):
        """
        Only used if DeepXube explicitly requests NumPy.
        Safe fallback that does NOT disable GPU mode.
        """
        tensors = self.to_torch(states, goals)
        return [t.detach().cpu().numpy() for t in tensors]
