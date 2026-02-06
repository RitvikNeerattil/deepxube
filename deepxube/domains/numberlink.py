from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import re

import numpy as np
from numpy.typing import NDArray

from deepxube.base.domain import State, Action, Goal, GoalStartRevWalkableActsRev, ActsEnumFixed
from deepxube.base.nnet_input import HasFlatSGActsEnumFixedIn, HasFlatSGAIn
from deepxube.base.factory import Parser
from deepxube.factories.domain_factory import domain_factory

from numberlink import GeneratorConfig, VariantConfig, RewardConfig, RenderConfig, NumberLinkRGBVectorEnv


class NumberLinkState(State):
    __slots__ = [
        "grid",
        "lane_v",
        "lane_h",
        "stack_rows",
        "stack_cols",
        "stack_lane",
        "stack_len",
        "heads",
        "closed",
        "step_count",
        "arm_presence",
        "hash",
    ]

    def __init__(
        self,
        *,
        grid: NDArray,
        lane_v: NDArray,
        lane_h: NDArray,
        stack_rows: NDArray,
        stack_cols: NDArray,
        stack_lane: NDArray,
        stack_len: NDArray,
        heads: NDArray,
        closed: NDArray,
        step_count: int,
        arm_presence: NDArray,
    ) -> None:
        self.grid = grid
        self.lane_v = lane_v
        self.lane_h = lane_h
        self.stack_rows = stack_rows
        self.stack_cols = stack_cols
        self.stack_lane = stack_lane
        self.stack_len = stack_len
        self.heads = heads
        self.closed = closed
        self.step_count = int(step_count)
        self.arm_presence = arm_presence
        self.hash: Optional[int] = None

    def __hash__(self) -> int:
        if self.hash is None:
            self.hash = hash(
                (
                    self.grid.tobytes(),
                    self.lane_v.tobytes(),
                    self.lane_h.tobytes(),
                    self.stack_rows.tobytes(),
                    self.stack_cols.tobytes(),
                    self.stack_lane.tobytes(),
                    self.stack_len.tobytes(),
                    self.heads.tobytes(),
                    self.closed.tobytes(),
                    self.step_count,
                    self.arm_presence.tobytes(),
                )
            )
        return self.hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NumberLinkState):
            return NotImplemented
        return (
            self.step_count == other.step_count
            and np.array_equal(self.grid, other.grid)
            and np.array_equal(self.lane_v, other.lane_v)
            and np.array_equal(self.lane_h, other.lane_h)
            and np.array_equal(self.stack_rows, other.stack_rows)
            and np.array_equal(self.stack_cols, other.stack_cols)
            and np.array_equal(self.stack_lane, other.stack_lane)
            and np.array_equal(self.stack_len, other.stack_len)
            and np.array_equal(self.heads, other.heads)
            and np.array_equal(self.closed, other.closed)
            and np.array_equal(self.arm_presence, other.arm_presence)
        )


class NumberLinkGoal(Goal):
    pass


class NumberLinkAction(Action):
    def __init__(self, action: int) -> None:
        self.action = int(action)

    def __hash__(self) -> int:
        return self.action

    def __eq__(self, other: object) -> bool:
        if isinstance(other, NumberLinkAction):
            return self.action == other.action
        return NotImplemented

    def __repr__(self) -> str:
        return f"{self.action}"


@domain_factory.register_class("numberlink")
class NumberLink(
    GoalStartRevWalkableActsRev[NumberLinkState, NumberLinkAction, NumberLinkGoal],
    ActsEnumFixed[NumberLinkState, NumberLinkAction, NumberLinkGoal],
    HasFlatSGActsEnumFixedIn[NumberLinkState, NumberLinkAction, NumberLinkGoal],
    HasFlatSGAIn[NumberLinkState, NumberLinkAction, NumberLinkGoal],
):
    def __init__(
        self,
        width: int = 8,
        height: int = 8,
        colors: int = 7,
        mode: str = "hamiltonian",
        seed: int | None = None,
        must_fill: bool = True,
        allow_diagonal: bool = False,
        bridges_probability: float = 0.0,
        cell_switching_mode: bool = False,
        level_id: str | None = None,
        step_limit: int | None = None,
    ) -> None:
        super().__init__()
        self.generator_cfg = GeneratorConfig(
            mode=mode,
            width=width,
            height=height,
            colors=colors,
            bridges_probability=bridges_probability,
            seed=seed,
        )
        self.variant_cfg = VariantConfig(
            must_fill=must_fill,
            allow_diagonal=allow_diagonal,
            bridges_enabled=bridges_probability > 0.0,
            cell_switching_mode=cell_switching_mode,
        )
        self.reward_cfg = RewardConfig()
        self.render_cfg = RenderConfig(show_endpoint_numbers=False, gridline_color=None)
        self.level_id = level_id
        self.step_limit = step_limit

        self._env = self._build_env(num_envs=1)
        self._goal_state = self._compute_goal_state()
        self._actions = [NumberLinkAction(i) for i in range(self._env.single_action_space.n)]

        self._height = self._env.H
        self._width = self._env.W
        self._num_colors = self._env.num_colors
        self._action_size = self._env.single_action_space.n
        self._actions_per_color = self._env._actions_per_color
        self._num_dirs = self._env._num_dirs

        self._dir_rev = self._compute_dir_reverse(self._env._dirs)

    def _build_env(self, num_envs: int) -> NumberLinkRGBVectorEnv:
        return NumberLinkRGBVectorEnv(
            num_envs=num_envs,
            level_id=self.level_id,
            generator=self.generator_cfg,
            variant=self.variant_cfg,
            reward_config=self.reward_cfg,
            render_config=self.render_cfg,
            step_limit=self.step_limit,
        )

    def _ensure_env(self, num_envs: int) -> NumberLinkRGBVectorEnv:
        if (self._env is None) or (self._env.num_envs != num_envs):
            self._env = self._build_env(num_envs=num_envs)
        return self._env

    @staticmethod
    def _compute_dir_reverse(dirs: NDArray[np.signedinteger]) -> List[int]:
        dir_to_idx: Dict[tuple[int, int], int] = {
            (int(vec[0]), int(vec[1])): int(idx) for idx, vec in enumerate(dirs)
        }
        rev: List[int] = []
        for vec in dirs:
            rev_idx = dir_to_idx.get((-int(vec[0]), -int(vec[1])))
            rev.append(int(rev_idx) if rev_idx is not None else 0)
        return rev

    def _compute_goal_state(self) -> NumberLinkState:
        env = self._build_env(num_envs=1)
        env.reset()
        solution = env.get_solution()
        if solution:
            for act in solution:
                env.step(np.array([act], dtype=np.int64))
        return self._capture_state(env, 0)

    @staticmethod
    def _capture_state(env: NumberLinkRGBVectorEnv, idx: int) -> NumberLinkState:
        return NumberLinkState(
            grid=env._grid_codes[idx].copy(),
            lane_v=env._lane_v[idx].copy(),
            lane_h=env._lane_h[idx].copy(),
            stack_rows=env._stack_rows[idx].copy(),
            stack_cols=env._stack_cols[idx].copy(),
            stack_lane=env._stack_lane[idx].copy(),
            stack_len=env._stack_len[idx].copy(),
            heads=env._heads[idx].copy(),
            closed=env._closed[idx].copy(),
            step_count=int(env._step_count[idx]),
            arm_presence=env._arm_presence[idx].copy(),
        )

    @staticmethod
    def _load_states(env: NumberLinkRGBVectorEnv, states: List[NumberLinkState]) -> None:
        for i, state in enumerate(states):
            env._grid_codes[i] = state.grid
            env._lane_v[i] = state.lane_v
            env._lane_h[i] = state.lane_h
            env._stack_rows[i] = state.stack_rows
            env._stack_cols[i] = state.stack_cols
            env._stack_lane[i] = state.stack_lane
            env._stack_len[i] = state.stack_len
            env._heads[i] = state.heads
            env._closed[i] = state.closed
            env._step_count[i] = state.step_count
            env._done_mask[i] = False
            env._arm_presence[i] = state.arm_presence

    def sample_goal_state_goal_pairs(self, num: int) -> Tuple[List[NumberLinkState], List[NumberLinkGoal]]:
        states_goal = [self._goal_state] * num
        goals = [NumberLinkGoal() for _ in range(num)]
        return states_goal, goals

    def sample_start_states(self, num: int) -> List[NumberLinkState]:
        env = self._ensure_env(num)
        env.reset()
        return [self._capture_state(env, i) for i in range(num)]

    def sample_start_goal_pairs(
        self, num_steps_l: List[int], times: Optional[Any] = None
    ) -> Tuple[List[NumberLinkState], List[NumberLinkGoal]]:
        # Forward-walk from random start states. The goal object is unused by is_solved.
        states_start = self.sample_start_states(len(num_steps_l))
        if any(step > 0 for step in num_steps_l):
            states_start = self.random_walk(states_start, num_steps_l)[0]
        goals = [NumberLinkGoal() for _ in range(len(num_steps_l))]
        return states_start, goals

    def is_solved(self, states: List[NumberLinkState], goals: List[NumberLinkGoal]) -> List[bool]:
        env = self._ensure_env(len(states))
        self._load_states(env, states)
        solved = env._compute_solved_mask()
        return solved.astype(bool).tolist()

    def get_actions_fixed(self) -> List[NumberLinkAction]:
        return self._actions.copy()

    def get_state_actions(self, states: List[NumberLinkState]) -> List[List[NumberLinkAction]]:
        env = self._ensure_env(len(states))
        self._load_states(env, states)
        mask = env._compute_action_mask()
        actions: List[List[NumberLinkAction]] = []
        for row in mask:
            valid = np.where(row != 0)[0]
            if valid.size == 0:
                actions.append([NumberLinkAction(0)])
            else:
                actions.append([NumberLinkAction(int(a)) for a in valid.tolist()])
        return actions

    def sample_state_action(self, states: List[NumberLinkState]) -> List[NumberLinkAction]:
        env = self._ensure_env(len(states))
        self._load_states(env, states)
        mask = env._compute_action_mask()
        actions: List[NumberLinkAction] = []
        for row in mask:
            valid = np.where(row != 0)[0]
            if valid.size == 0:
                actions.append(NumberLinkAction(0))
            else:
                actions.append(NumberLinkAction(int(np.random.choice(valid))))
        return actions

    def next_state(
        self, states: List[NumberLinkState], actions: List[NumberLinkAction]
    ) -> Tuple[List[NumberLinkState], List[float]]:
        env = self._ensure_env(len(states))
        self._load_states(env, states)
        acts = np.array([a.action for a in actions], dtype=np.int64)
        env.step(acts)
        next_states = [self._capture_state(env, i) for i in range(len(states))]
        return next_states, [1.0] * len(states)

    def actions_to_indices(self, actions: List[NumberLinkAction]) -> List[int]:
        return [a.action for a in actions]

    def get_input_info_flat_sg(self) -> Tuple[List[int], List[int]]:
        cell_count = self._height * self._width
        depth = self._num_colors + 1
        return [cell_count, cell_count, cell_count], [depth, depth, depth]

    def get_input_info_flat_sga(self) -> Tuple[List[int], List[int]]:
        cell_count = self._height * self._width
        depth = self._num_colors + 1
        return [cell_count, cell_count, cell_count, 1], [depth, depth, depth, self._action_size]

    def to_np_flat_sg(self, states: List[NumberLinkState], goals: List[NumberLinkGoal]) -> List[NDArray]:
        grid = np.stack([s.grid.reshape(-1) for s in states], axis=0).astype(np.uint8)
        lane_v = np.stack([s.lane_v.reshape(-1) for s in states], axis=0).astype(np.uint8)
        lane_h = np.stack([s.lane_h.reshape(-1) for s in states], axis=0).astype(np.uint8)
        return [grid, lane_v, lane_h]

    def to_np_flat_sga(
        self, states: List[NumberLinkState], goals: List[NumberLinkGoal], actions: List[NumberLinkAction]
    ) -> List[NDArray]:
        return self.to_np_flat_sg(states, goals) + [np.expand_dims(np.array(self.actions_to_indices(actions)), 1)]

    def rev_action(self, states: List[NumberLinkState], actions: List[NumberLinkAction]) -> List[NumberLinkAction]:
        if self.variant_cfg.cell_switching_mode:
            return actions
        actions_rev: List[NumberLinkAction] = []
        for action in actions:
            action_idx = action.action
            color_idx = action_idx // self._actions_per_color
            remainder = action_idx % self._actions_per_color
            head_idx = remainder // self._num_dirs
            dir_idx = remainder % self._num_dirs
            rev_dir = self._dir_rev[dir_idx]
            rev_action = color_idx * self._actions_per_color + head_idx * self._num_dirs + rev_dir
            actions_rev.append(NumberLinkAction(int(rev_action)))
        return actions_rev

    def __repr__(self) -> str:
        return (
            "NumberLink("
            f"width={self.generator_cfg.width}, "
            f"height={self.generator_cfg.height}, "
            f"colors={self.generator_cfg.colors}, "
            f"mode={self.generator_cfg.mode}, "
            f"cell_switching_mode={self.variant_cfg.cell_switching_mode})"
        )


@domain_factory.register_parser("numberlink")
class NumberLinkParser(Parser):
    _pattern = re.compile(
        r"^(?P<w>\d+)x(?P<h>\d+)x(?P<c>\d+)(?:_(?P<mode>hamiltonian|random_walk))?(?:_seed(?P<seed>-?\d+))?$"
    )

    def parse(self, args_str: str) -> Dict[str, Any]:
        if args_str.startswith("level="):
            return {"level_id": args_str.split("=", 1)[1]}
        match = self._pattern.match(args_str)
        if match is None:
            raise ValueError("Invalid format")
        groups = match.groupdict()
        kwargs: Dict[str, Any] = {
            "width": int(groups["w"]),
            "height": int(groups["h"]),
            "colors": int(groups["c"]),
        }
        if groups.get("mode"):
            kwargs["mode"] = groups["mode"]
        if groups.get("seed") is not None:
            kwargs["seed"] = int(groups["seed"])
        return kwargs

    def help(self) -> str:
        return (
            "Formats:\n"
            "  <width>x<height>x<colors>[_<mode>][_seed<seed>]\n"
            "  level=<level_id>\n"
            "Examples: numberlink.8x8x7, numberlink.10x8x6_random_walk_seed42, numberlink.level=nl_easy_1"
        )
