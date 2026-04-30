from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, TYPE_CHECKING
import re

import numpy as np
from numpy.typing import NDArray

from deepxube.base.domain import (
    State,
    Action,
    Goal,
    GoalStartRevWalkableActsRev,
    ActsEnumFixed,
    StateGoalVizable,
    StringToAct,
)
from deepxube.base.nnet_input import HasFlatSGActsEnumFixedIn, HasFlatSGAIn
from deepxube.base.factory import Parser
from deepxube.factories.domain_factory import domain_factory

from numberlink import GeneratorConfig, VariantConfig, RewardConfig, RenderConfig, NumberLinkRGBVectorEnv

if TYPE_CHECKING:
    from matplotlib.figure import Figure


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
    StateGoalVizable[NumberLinkState, NumberLinkAction, NumberLinkGoal],
    StringToAct[NumberLinkState, NumberLinkAction, NumberLinkGoal],
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
        self.render_cfg = RenderConfig(
            show_endpoint_numbers=False,
            gridline_color=None,
            endpoint_border_thickness=0,
            connection_color_adjustment=0,
        )
        self.level_id = level_id
        self.step_limit = step_limit

        self._env = self._build_env(num_envs=1)
        self._expand_env: Optional[NumberLinkRGBVectorEnv] = None
        self._height = self._env.H
        self._width = self._env.W
        self._num_colors = self._env.num_colors
        self._action_size = self._env.single_action_space.n
        self._actions_per_color = self._env._actions_per_color
        self._num_dirs = self._env._num_dirs
        self._bridges_mask = self._env._bridges_mask.copy()
        self._non_bridge_mask = ~self._bridges_mask
        self._goal_state = self._compute_goal_state()
        self._actions = [NumberLinkAction(i) for i in range(self._env.single_action_space.n)]
        self._viz_legal_action_ids: set[int] = set()
        self._viz_head_choices: Dict[int, tuple[int, int]] = {}
        self._viz_selected_choice: Optional[int] = None
        self._viz_selected_head: Optional[tuple[int, int]] = None
        self._viz_input_handled: bool = False
        self._viz_status: str = "Select a head number, then type up/down/left/right."
        self._color_names = [
            "red",
            "green",
            "blue",
            "orange",
            "purple",
            "cyan",
            "magenta",
            "pink",
            "teal",
            "lavender",
            "brown",
            "beige",
            "maroon",
            "mint",
            "olive",
            "coral",
            "navy",
            "grey",
            "yellow",
            "aqua",
        ]

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

    def _ensure_expand_env(self, num_envs: int) -> NumberLinkRGBVectorEnv:
        if (self._expand_env is None) or (self._expand_env.num_envs != num_envs):
            self._expand_env = self._build_env(num_envs=num_envs)
        return self._expand_env

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
        state = self._capture_state(env, 0)
        if self._state_is_solved_path_mode(state):
            return state

        solution = env.get_solution()
        if solution:
            for act in solution:
                env.step(np.array([act], dtype=np.int64))
                state = self._capture_state(env, 0)
                if self._state_is_solved_path_mode(state):
                    return state

        # Fallback: return final state if solved state was not observed.
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
        # Curriculum-friendly starts: apply a prefix of a known solution to get near-solved states.
        # Falls back to random starts if no solution is available for the generated level.
        states_start: List[NumberLinkState] = [self._goal_state] * len(num_steps_l)
        goals = [NumberLinkGoal() for _ in range(len(num_steps_l))]

        step_to_idxs: Dict[int, List[int]] = {}
        for idx, step in enumerate(num_steps_l):
            step_to_idxs.setdefault(int(step), []).append(idx)

        for step_count, idxs in step_to_idxs.items():
            env = self._build_env(num_envs=len(idxs))
            env.reset()
            solution = env.get_solution()
            # Capture trajectory snapshots and select the prefix aligned to first solved step.
            snapshots: Dict[int, List[NumberLinkState]] = {
                0: [self._capture_state(env, i) for i in range(len(idxs))]
            }
            solved_step: Optional[int] = 0 if self._state_is_solved_path_mode(snapshots[0][0]) else None

            if solution:
                for step_idx, act in enumerate(solution, start=1):
                    env.step(np.full((len(idxs),), int(act), dtype=np.int64))
                    snapshots[step_idx] = [self._capture_state(env, i) for i in range(len(idxs))]
                    if (solved_step is None) and self._state_is_solved_path_mode(snapshots[step_idx][0]):
                        solved_step = step_idx

            if solved_step is None:
                solved_step = max(snapshots.keys())

            steps_remaining = min(int(step_count), solved_step)
            prefix_len = solved_step - steps_remaining
            selected_states = snapshots.get(prefix_len, snapshots[max(snapshots.keys())])
            for pos, state in enumerate(selected_states):
                states_start[idxs[pos]] = state

        return states_start, goals

    def is_solved(self, states: List[NumberLinkState], goals: List[NumberLinkGoal]) -> List[bool]:
        if len(states) == 0:
            return []

        # Fast path mode check: closed connections plus fill/region constraints.
        # This avoids loading states into the vector env for every query.
        if not self.variant_cfg.cell_switching_mode:
            return [self._state_is_solved_path_mode(state) for state in states]

        env = self._ensure_env(len(states))
        self._load_states(env, states)
        solved = env._compute_solved_mask()
        return solved.astype(bool).tolist()

    def _state_is_solved_path_mode(self, state: NumberLinkState) -> bool:
        if self.variant_cfg.must_fill:
            normal_ok = bool(np.all((state.grid != 0) | self._bridges_mask))
            bridge_ok = bool(np.all(self._non_bridge_mask | ((state.lane_v != 0) | (state.lane_h != 0))))
            if not (normal_ok and bridge_ok):
                return False
        return (
            self._colors_have_single_regions(state)
            and self._colors_match_path_stacks(state)
            and self._path_stacks_are_continuous(state)
        )

    def _color_mask(self, state: NumberLinkState, color_code: int) -> NDArray[np.bool_]:
        color_mask = (state.grid == color_code) & self._non_bridge_mask
        if np.any(self._bridges_mask):
            color_mask = color_mask | (
                self._bridges_mask & ((state.lane_v == color_code) | (state.lane_h == color_code))
            )
        return color_mask

    def _stack_mask(self, state: NumberLinkState, color_idx: int) -> NDArray[np.bool_]:
        stack_mask = np.zeros((self._height, self._width), dtype=np.bool_)
        for head_idx in range(2):
            stack_len = int(state.stack_len[color_idx, head_idx])
            if stack_len <= 0:
                continue
            rows = state.stack_rows[color_idx, head_idx, :stack_len].astype(np.intp, copy=False)
            cols = state.stack_cols[color_idx, head_idx, :stack_len].astype(np.intp, copy=False)
            stack_mask[rows, cols] = True
        return stack_mask

    def _colors_match_path_stacks(self, state: NumberLinkState) -> bool:
        for color_idx in range(self._num_colors):
            color_mask = self._color_mask(state, color_idx + 1)
            stack_mask = self._stack_mask(state, color_idx)
            if bool(np.any(color_mask != stack_mask)):
                return False
        return True

    def _path_stacks_are_continuous(self, state: NumberLinkState) -> bool:
        for color_idx in range(self._num_colors):
            for head_idx in range(2):
                stack_len = int(state.stack_len[color_idx, head_idx])
                if stack_len <= 0:
                    return False
                rows = state.stack_rows[color_idx, head_idx, :stack_len].astype(np.int64, copy=False)
                cols = state.stack_cols[color_idx, head_idx, :stack_len].astype(np.int64, copy=False)
                if stack_len > 1:
                    dists = np.abs(np.diff(rows)) + np.abs(np.diff(cols))
                    if bool(np.any(dists != 1)):
                        return False

            if bool(state.closed[color_idx]):
                head0 = state.heads[color_idx, 0].astype(np.int64, copy=False)
                head1 = state.heads[color_idx, 1].astype(np.int64, copy=False)
                if int(np.abs(head0[0] - head1[0]) + np.abs(head0[1] - head1[1])) > 1:
                    return False
        return True

    def _color_stack_statuses(self, state: NumberLinkState) -> List[str]:
        statuses: List[str] = []
        for color_idx in range(self._num_colors):
            color_mask = self._color_mask(state, color_idx + 1)
            stack_mask = self._stack_mask(state, color_idx)
            extra_grid = int(np.sum(color_mask & ~stack_mask))
            extra_stack = int(np.sum(stack_mask & ~color_mask))
            if extra_grid == 0 and extra_stack == 0:
                statuses.append("ok")
            else:
                statuses.append(f"bad(+grid {extra_grid}, +stack {extra_stack})")
        return statuses

    def _color_stack_continuity_statuses(self, state: NumberLinkState) -> List[str]:
        statuses: List[str] = []
        for color_idx in range(self._num_colors):
            ok = True
            for head_idx in range(2):
                stack_len = int(state.stack_len[color_idx, head_idx])
                if stack_len <= 0:
                    ok = False
                    break
                rows = state.stack_rows[color_idx, head_idx, :stack_len].astype(np.int64, copy=False)
                cols = state.stack_cols[color_idx, head_idx, :stack_len].astype(np.int64, copy=False)
                if stack_len > 1:
                    dists = np.abs(np.diff(rows)) + np.abs(np.diff(cols))
                    if bool(np.any(dists != 1)):
                        ok = False
                        break
            if ok and bool(state.closed[color_idx]):
                head0 = state.heads[color_idx, 0].astype(np.int64, copy=False)
                head1 = state.heads[color_idx, 1].astype(np.int64, copy=False)
                ok = int(np.abs(head0[0] - head1[0]) + np.abs(head0[1] - head1[1])) <= 1
            statuses.append("ok" if ok else "bad")
        return statuses

    def _computed_color_statuses(self, state: NumberLinkState) -> List[str]:
        region_counts = self._color_region_counts(state)
        stack_statuses = self._color_stack_statuses(state)
        continuity_statuses = self._color_stack_continuity_statuses(state)
        statuses: List[str] = []
        for idx in range(self._num_colors):
            ok = region_counts[idx] == 1 and stack_statuses[idx] == "ok" and continuity_statuses[idx] == "ok"
            statuses.append("ok" if ok else "bad")
        return statuses

    def _color_region_counts(self, state: NumberLinkState) -> List[int]:
        region_counts: List[int] = []
        for color_code in range(1, self._num_colors + 1):
            color_mask = self._color_mask(state, color_code)

            coords = np.argwhere(color_mask)
            if coords.size == 0:
                region_counts.append(0)
                continue

            visited = np.zeros_like(color_mask, dtype=np.bool_)
            count = 0
            for row_raw, col_raw in coords:
                row = int(row_raw)
                col = int(col_raw)
                if visited[row, col]:
                    continue
                count += 1
                stack: List[tuple[int, int]] = [(row, col)]
                visited[row, col] = True
                while stack:
                    row_curr, col_curr = stack.pop()
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        row_next = row_curr + dr
                        col_next = col_curr + dc
                        if not (0 <= row_next < self._height and 0 <= col_next < self._width):
                            continue
                        if visited[row_next, col_next] or not color_mask[row_next, col_next]:
                            continue
                        visited[row_next, col_next] = True
                        stack.append((row_next, col_next))
            region_counts.append(count)
        return region_counts

    def _colors_have_single_regions(self, state: NumberLinkState) -> bool:
        return all(count == 1 for count in self._color_region_counts(state))

    def get_actions_fixed(self) -> List[NumberLinkAction]:
        return self._actions.copy()

    def _color_name(self, color_idx: int) -> str:
        if color_idx < len(self._color_names):
            return self._color_names[color_idx]
        return f"color {color_idx + 1}"

    def _coord_to_xy(self, row: int, col: int) -> tuple[int, int]:
        return int(col), int(self._height - 1 - row)

    def _decode_path_action(self, action_idx: int) -> tuple[int, int, int, str]:
        color_idx = action_idx // self._actions_per_color
        remainder = action_idx % self._actions_per_color
        head_idx = remainder // self._num_dirs
        dir_idx = remainder % self._num_dirs
        dr, dc = self._env._dirs[dir_idx]
        dir_name = {
            (-1, 0): "up",
            (1, 0): "down",
            (0, -1): "left",
            (0, 1): "right",
        }.get((int(dr), int(dc)), f"({int(dr)},{int(dc)})")
        return color_idx, head_idx, dir_idx, dir_name

    def _action_to_str(self, action_idx: int) -> str:
        if self.variant_cfg.cell_switching_mode:
            row = action_idx // (self._width * (self._num_colors + 1))
            rem = action_idx % (self._width * (self._num_colors + 1))
            col = rem // (self._num_colors + 1)
            color = rem % (self._num_colors + 1)
            color_text = "empty" if color == 0 else self._color_name(color - 1)
            x, y = self._coord_to_xy(row, col)
            return f"{action_idx}: set cell ({x},{y}) to {color_text}"

        color_idx, head_idx, _, dir_name = self._decode_path_action(action_idx)
        if hasattr(self, "_viz_state_for_action_labels"):
            head_row, head_col = self._viz_state_for_action_labels.heads[color_idx, head_idx]
            x, y = self._coord_to_xy(int(head_row), int(head_col))
            head_text = f"head {head_idx} at ({x},{y})"
        else:
            head_text = f"head {head_idx}"
        return f"{action_idx}: move {self._color_name(color_idx)} {head_text} {dir_name}"

    def _set_viz_head_choices(self, legal_actions: List[NumberLinkAction]) -> None:
        head_to_dirs: Dict[tuple[int, int], List[str]] = {}
        for action in legal_actions:
            color_idx, head_idx, _, dir_name = self._decode_path_action(action.action)
            head_to_dirs.setdefault((color_idx, head_idx), []).append(dir_name)

        self._viz_head_choices = {
            choice_idx: head for choice_idx, head in enumerate(sorted(head_to_dirs.keys()), start=1)
        }
        if self._viz_selected_head not in head_to_dirs:
            self._viz_selected_choice = 1 if self._viz_head_choices else None
            self._viz_selected_head = self._viz_head_choices.get(1)
        else:
            self._viz_selected_choice = next(
                choice_idx
                for choice_idx, head in self._viz_head_choices.items()
                if head == self._viz_selected_head
            )

    def _viz_head_lines(self, legal_actions: List[NumberLinkAction]) -> List[str]:
        if self.variant_cfg.cell_switching_mode:
            return ["Cell switching mode: type an action number."]

        head_to_dirs: Dict[tuple[int, int], List[str]] = {}
        for action in legal_actions:
            color_idx, head_idx, _, dir_name = self._decode_path_action(action.action)
            head_to_dirs.setdefault((color_idx, head_idx), []).append(dir_name)

        lines: List[str] = []
        for choice_idx, (color_idx, head_idx) in self._viz_head_choices.items():
            head_row, head_col = self._viz_state_for_action_labels.heads[color_idx, head_idx]
            marker = "*" if self._viz_selected_choice == choice_idx else " "
            dirs = "/".join(sorted(head_to_dirs.get((color_idx, head_idx), [])))
            x, y = self._coord_to_xy(int(head_row), int(head_col))
            lines.append(
                f"{marker}{choice_idx}: {self._color_name(color_idx)} head {head_idx} "
                f"at ({x},{y}) [{dirs}]"
            )
        return lines

    def _draw_viz_markers(self, state: NumberLinkState, frame: NDArray, ax_img: Any) -> None:
        from matplotlib.patches import Circle, Rectangle

        pix_h = frame.shape[0] / self._height
        pix_w = frame.shape[1] / self._width
        self._viz_head_marker_centers: Dict[int, tuple[float, float, float]] = {}

        # Fixed start endpoints: outline only, no numbers, because these do not move.
        for color_idx in range(self._num_colors):
            for head_idx in range(2):
                row = int(state.stack_rows[color_idx, head_idx, 0])
                col = int(state.stack_cols[color_idx, head_idx, 0])
                x = (col * pix_w) - 0.5
                y = (row * pix_h) - 0.5
                ax_img.add_patch(
                    Rectangle((x, y), pix_w, pix_h, fill=False, edgecolor="black", linewidth=3.0)
                )
                ax_img.add_patch(
                    Rectangle((x, y), pix_w, pix_h, fill=False, edgecolor="white", linewidth=1.5)
                )

        # Active selectable heads: numbered to match the "Heads:" choice list and move after each action.
        radius = max(0.28, min(pix_w, pix_h) * 0.28)
        for choice_idx, (color_idx, head_idx) in self._viz_head_choices.items():
            row, col = state.heads[color_idx, head_idx]
            center_x = int(col) * pix_w + (pix_w - 1.0) / 2.0
            center_y = int(row) * pix_h + (pix_h - 1.0) / 2.0
            selected = self._viz_selected_choice == choice_idx
            self._viz_head_marker_centers[choice_idx] = (float(center_x), float(center_y), float(radius * 1.5))
            ax_img.add_patch(
                Circle(
                    (center_x, center_y),
                    radius=radius,
                    facecolor="yellow" if selected else "white",
                    edgecolor="black",
                    linewidth=1.5,
                    alpha=0.9,
                )
            )
            ax_img.text(
                center_x,
                center_y,
                str(choice_idx),
                ha="center",
                va="center",
                color="black",
                fontsize=8,
                weight="bold",
            )
            if selected:
                ax_img.add_patch(
                    Circle((center_x, center_y), radius=radius * 1.35, fill=False, edgecolor="yellow", linewidth=2.0)
                )

    def visualize_state_goal(self, state: NumberLinkState, goal: NumberLinkGoal, fig: "Figure") -> None:
        env = self._ensure_env(1)
        self._load_states(env, [state])
        frames = env.render()
        if frames is None:
            raise ValueError("NumberLink rendering is unavailable for this configuration")

        legal_actions = self.get_state_actions([state])[0]
        self._viz_legal_action_ids = {action.action for action in legal_actions}
        self._viz_state_for_action_labels = state
        self._set_viz_head_choices(legal_actions)
        legal_action_lines = [self._action_to_str(action.action) for action in legal_actions[:18]]
        if len(legal_actions) > 18:
            legal_action_lines.append(f"... {len(legal_actions) - 18} more")

        fig.clear()
        ax_img = fig.add_axes((0.04, 0.12, 0.42, 0.78))
        ax_info = fig.add_axes((0.51, 0.08, 0.46, 0.86))

        ax_img.imshow(frames[0])
        self._viz_ax_img = ax_img
        ax_img.set_title("NumberLink state")
        ax_img.set_aspect("equal", adjustable="box")
        self._draw_viz_markers(state, frames[0], ax_img)
        ax_img.axis("off")

        solved = self.is_solved([state], [goal])[0]
        region_counts = ", ".join(
            f"{self._color_name(idx)}={count}"
            for idx, count in enumerate(self._color_region_counts(state))
        )
        stack_statuses = ", ".join(
            f"{self._color_name(idx)}={status}"
            for idx, status in enumerate(self._color_stack_statuses(state))
        )
        continuity_statuses = ", ".join(
            f"{self._color_name(idx)}={status}"
            for idx, status in enumerate(self._color_stack_continuity_statuses(state))
        )
        computed_statuses = ", ".join(
            f"{self._color_name(idx)}={status}"
            for idx, status in enumerate(self._computed_color_statuses(state))
        )
        info = [
            f"Solved: {solved}",
            f"Step count: {state.step_count}",
            f"Env closed flags: {int(np.sum(state.closed))}/{len(state.closed)}",
            f"Computed colors: {computed_statuses}",
            f"Color regions: {region_counts}",
            f"Path stacks: {stack_statuses}",
            f"Continuity: {continuity_statuses}",
            f"Legal actions: {len(legal_actions)}",
            "Endpoint outlines: fixed start cells.",
            "Head circles: numbered to match the selectable Heads list.",
            "",
            self._viz_status,
            "",
            "Heads:",
            *self._viz_head_lines(legal_actions),
            "",
            "Raw legal actions:",
            *legal_action_lines,
        ]
        ax_info.text(0.0, 1.0, "\n".join(info), va="top", ha="left", family="monospace", fontsize=9)
        ax_info.axis("off")

    def viz_select_head_at(self, x: float, y: float) -> bool:
        if x is None or y is None:
            return False

        for choice_idx, (center_x, center_y, radius) in getattr(self, "_viz_head_marker_centers", {}).items():
            if ((float(x) - center_x) ** 2 + (float(y) - center_y) ** 2) <= radius**2:
                self._select_viz_head(choice_idx)
                return True
        return False

    def string_to_action(self, act_str: str) -> Optional[NumberLinkAction]:
        self._viz_input_handled = False
        act_str_norm = act_str.strip().lower()
        dir_aliases = {
            "u": "up",
            "up": "up",
            "d": "down",
            "down": "down",
            "l": "left",
            "left": "left",
            "r": "right",
            "right": "right",
        }

        parts = act_str_norm.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1] in dir_aliases:
            self._select_viz_head(int(parts[0]))
            return self._viz_direction_to_action(dir_aliases[parts[1]])

        if act_str_norm.isdigit():
            self._select_viz_head(int(act_str_norm))
            return None

        if act_str_norm in dir_aliases:
            return self._viz_direction_to_action(dir_aliases[act_str_norm])

        try:
            action_idx = int(act_str)
        except ValueError:
            return None

        if action_idx in self._viz_legal_action_ids:
            return self._actions[action_idx]
        return None

    def string_to_action_help(self) -> str:
        return (
            "Type 1, 2, 3, ... to select a listed color head, then type up/down/left/right "
            "(or u/d/l/r). You can also type '<head> <direction>', e.g. '2 left'."
        )

    def _select_viz_head(self, choice_idx: int) -> None:
        self._viz_input_handled = True
        selected = self._viz_head_choices.get(choice_idx)
        if selected is None:
            self._viz_status = f"No head choice {choice_idx}. Choose one listed under Heads."
            return

        self._viz_selected_head = selected
        self._viz_selected_choice = choice_idx
        color_idx, head_idx = selected
        self._viz_status = (
            f"Selected choice {choice_idx}: {self._color_name(color_idx)} head {head_idx}. Type a direction."
        )

    def _viz_direction_to_action(self, dir_name: str) -> Optional[NumberLinkAction]:
        self._viz_input_handled = True
        if self._viz_selected_head is None:
            self._viz_status = "Select a head number first."
            return None

        selected_color, selected_head = self._viz_selected_head
        selected_choice = self._viz_selected_choice
        for action_idx in sorted(self._viz_legal_action_ids):
            color_idx, head_idx, _, action_dir = self._decode_path_action(action_idx)
            if (color_idx, head_idx, action_dir) == (selected_color, selected_head, dir_name):
                choice_text = f"choice {selected_choice}: " if selected_choice is not None else ""
                self._viz_status = (
                    f"Moved {choice_text}{self._color_name(color_idx)} head {head_idx} {dir_name} "
                    f"(action {action_idx})."
                )
                return self._actions[action_idx]

        self._viz_status = f"{dir_name} is not legal for selected choice {selected_choice}."
        return None

    def viz_selected_head_position(self, state: NumberLinkState) -> Optional[tuple[int, int]]:
        if self._viz_selected_head is None:
            return None
        color_idx, head_idx = self._viz_selected_head
        row, col = state.heads[color_idx, head_idx]
        return int(row), int(col)

    def viz_input_handled(self) -> bool:
        return self._viz_input_handled

    def get_state_actions(self, states: List[NumberLinkState]) -> List[List[NumberLinkAction]]:
        env = self._ensure_env(len(states))
        self._load_states(env, states)
        mask = env._compute_action_mask()
        actions: List[List[NumberLinkAction]] = []
        for row in mask:
            valid = np.where(row != 0)[0]
            if valid.size == 0:
                actions.append([self._actions[0]])
            else:
                actions.append([self._actions[int(a)] for a in valid.tolist()])
        return actions

    def sample_state_action(self, states: List[NumberLinkState]) -> List[NumberLinkAction]:
        env = self._ensure_env(len(states))
        self._load_states(env, states)
        mask = env._compute_action_mask()
        actions: List[NumberLinkAction] = []
        for row in mask:
            valid = np.where(row != 0)[0]
            if valid.size == 0:
                actions.append(self._actions[0])
            else:
                actions.append(self._actions[int(np.random.choice(valid))])
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
