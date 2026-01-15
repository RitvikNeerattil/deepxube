from typing import List, Tuple, Optional, Any

import numpy as np
import random
from numpy.typing import NDArray
import torch
from torch import Tensor
import torch.nn as nn

from deepxube.base.env import Action, EnvEnumerableActs, EnvStartGoalRW, Goal, State
from deepxube.base.heuristic import HeurNNetModule, HeurNNetV
from deepxube.nnet.pytorch_models import Conv2dModel, FullyConnectedModel
from numberlink.config import GeneratorConfig, VariantConfig
from numberlink.vector_env import NumberLinkRGBVectorEnv


class NumberLinkState(State):
    """State for the NumberLink environment."""

    def __init__(self, grid: Any, grid_codes: NDArray[np.uint8], lane_v: NDArray[np.uint8], lane_h: NDArray[np.uint8],
                 closed: NDArray[np.bool_], steps: int):
        self.grid = grid
        self.grid_codes = grid_codes
        self.lane_v = lane_v
        self.lane_h = lane_h
        self.closed = closed
        self.steps = steps
        self._hash: Optional[int] = None

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash((self.grid_codes.tobytes(), self.lane_v.tobytes(), self.lane_h.tobytes(), self.closed.tobytes()))
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NumberLinkState):
            return False
        return (np.array_equal(self.grid_codes, other.grid_codes) and
                np.array_equal(self.lane_v, other.lane_v) and
                np.array_equal(self.lane_h, other.lane_h) and
                np.array_equal(self.closed, other.closed))


class NumberLinkAction(Action):
    """Action for the NumberLink environment."""

    def __init__(self, action: int):
        self.action = action

    def __hash__(self) -> int:
        return hash(self.action)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NumberLinkAction):
            return False
        return self.action == other.action

    def __repr__(self) -> str:
        return f"NumberLinkAction({self.action})"


class NumberLinkGoal(Goal):
    """Goal for the NumberLink environment."""
    pass


class NumberLinkDeepXubeEnv(EnvStartGoalRW[NumberLinkState, NumberLinkAction, NumberLinkGoal], EnvEnumerableActs):
    def __init__(self, width: int = 7, height: int = 7, num_colors: int = 5):
        super().__init__()
        self.width = width
        self.height = height
        self.num_colors = num_colors

        # Create a template env to get parameters
        self.template_env = NumberLinkRGBVectorEnv(
            num_envs=1,
            generator=GeneratorConfig(width=width, height=height, colors=num_colors),
            variant=VariantConfig(cell_switching_mode=True)
        )
        self.action_size = self.template_env.single_action_space.n

    def get_start_states(self, num_states: int) -> List[NumberLinkState]:
        env = NumberLinkRGBVectorEnv(
            num_envs=num_states,
            generator=GeneratorConfig(width=self.width, height=self.height, colors=self.num_colors),
            variant=VariantConfig(cell_switching_mode=True)
        )
        obs, info = env.reset()
        
        states = []
        for i in range(num_states):
            state = NumberLinkState(
                grid=env._grid[i],
                grid_codes=env._grid_codes[i],
                lane_v=env._lane_v[i],
                lane_h=env._lane_h[i],
                closed=env._closed[i],
                steps=env._step_count[i]
            )
            states.append(state)
        return states

    def sample_goal(self, states_start: List[NumberLinkState], states_goal: List[NumberLinkState]) -> List[NumberLinkGoal]:
        return [NumberLinkGoal() for _ in states_start]

    def get_state_actions(self, states: List[NumberLinkState]) -> List[List[NumberLinkAction]]:
        actions: List[List[NumberLinkAction]] = []
        # Create a temporary env to get action mask
        env = self._create_env_from_states(states)
        action_masks = env._compute_action_mask()

        for i in range(len(states)):
            valid_actions_indices = np.flatnonzero(action_masks[i])
            actions.append([NumberLinkAction(int(a)) for a in valid_actions_indices])
        return actions

    def _create_env_from_states(self, states: List[NumberLinkState]) -> NumberLinkRGBVectorEnv:
        num_envs = len(states)
        env = NumberLinkRGBVectorEnv(
            num_envs=num_envs,
            generator=GeneratorConfig(width=self.width, height=self.height, colors=self.num_colors),
            variant=VariantConfig(cell_switching_mode=True)
        )

        env._grid = [s.grid for s in states]
        for i, state in enumerate(states):
            env._grid_codes[i] = state.grid_codes
            env._lane_v[i] = state.lane_v
            env._lane_h[i] = state.lane_h
            env._closed[i] = state.closed
            env._step_count[i] = state.steps
        
        return env

    def next_state(self, states: List[NumberLinkState], actions: List[NumberLinkAction]) -> Tuple[List[NumberLinkState], List[float]]:
        env = self._create_env_from_states(states)
        
        np_actions = np.array([a.action for a in actions], dtype=int)
        
        obs, rewards, terminated, truncated, info = env.step(np_actions)
        
        next_states = []
        for i in range(len(states)):
            next_states.append(NumberLinkState(
                grid=env._grid[i],
                grid_codes=env._grid_codes[i],
                lane_v=env._lane_v[i],
                lane_h=env._lane_h[i],
                closed=env._closed[i],
                steps=env._step_count[i]
            ))
            
        return next_states, [-r for r in rewards]

    def is_solved(self, states: List[NumberLinkState], goals: List[NumberLinkGoal]) -> List[bool]:
        env = self._create_env_from_states(states)
        return list(env._compute_solved_mask())

    def get_state_action_rand(self, states: List[NumberLinkState]) -> List[NumberLinkAction]:
        state_actions_l = self.get_state_actions(states)
        return [random.choice(state_actions) if state_actions else NumberLinkAction(0) for state_actions in state_actions_l]

class NumberLinkNNet(HeurNNetModule):
    def __init__(self, width: int, height: int, num_colors: int, device: str = "cpu"):
        super().__init__()
        
        # The input will have 4 channels: grid_codes, lane_v, lane_h, closed
        input_channels = 4
        
        self.conv_layers = Conv2dModel(
            chan_in=input_channels,
            channel_sizes=[32, 64],
            kernel_sizes=[3, 3],
            paddings=[1, 1],
            layer_acts=["RELU", "RELU"],
            batch_norms=[True, True]
        )
        
        # Calculate the size of the flattened features after conv layers
        conv_output_size = self._get_conv_output_size(width, height, input_channels)

        self.fc_layers = FullyConnectedModel(
            input_dim=conv_output_size,
            dims=[128, 1],
            acts=["RELU", "LINEAR"]
        )
        self.to(device)

    def _get_conv_output_size(self, width: int, height: int, channels: int) -> int:
        # Helper to calculate the output size of the conv layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, channels, height, width)
            output = self.conv_layers(dummy_input)
            return output.flatten(1).shape[1]

    def forward(self, states_goals_l: List[Tensor]) -> Tensor:
        # states_goals_l[0] is the state tensor
        state_tensor = states_goals_l[0]
        
        # The input is expected to be (batch, channels, height, width)
        x = self.conv_layers(state_tensor)
        x = x.flatten(1)
        x = self.fc_layers(x)
        return x

class NumberLinkNNetParV(HeurNNetV[NumberLinkState, NumberLinkGoal]):
    def __init__(self, width: int, height: int, num_colors: int, device: str = "cpu"):
        super().__init__()
        self.width = width
        self.height = height
        self.num_colors = num_colors
        self.device = device

    def get_nnet(self) -> HeurNNetModule:
        return NumberLinkNNet(self.width, self.height, self.num_colors, self.device)

    def to_np(self, states: List[NumberLinkState], goals: List[NumberLinkGoal]) -> List[NDArray[Any]]:
        # Stack the grid representations from each state
        # The state is represented by grid_codes, lane_v, and lane_h
        # We stack them as channels of an image-like tensor
        
        batch_size = len(states)
        
        # Initialize numpy arrays for each channel
        grid_codes_np = np.zeros((batch_size, self.height, self.width), dtype=np.float32)
        lane_v_np = np.zeros((batch_size, self.height, self.width), dtype=np.float32)
        lane_h_np = np.zeros((batch_size, self.height, self.width), dtype=np.float32)
        closed_np = np.zeros((batch_size, self.height, self.width), dtype=np.float32)
        
        for i, state in enumerate(states):
            grid_codes_np[i] = state.grid_codes
            lane_v_np[i] = state.lane_v
            lane_h_np[i] = state.lane_h

            closed_channel = np.zeros((self.height, self.width), dtype=np.float32)
            for color_idx, is_closed in enumerate(state.closed):
                if is_closed:
                    color_code = color_idx + 1
                    closed_channel[state.grid_codes == color_code] = 1.0
                    closed_channel[state.lane_v == color_code] = 1.0
                    closed_channel[state.lane_h == color_code] = 1.0
            closed_np[i] = closed_channel

        # Stack the channels to form a (batch, channels, height, width) tensor
        states_np = np.stack([grid_codes_np, lane_v_np, lane_h_np, closed_np], axis=1)
        
        # Goals are not used in this simple V-function network, but the interface requires it.
        # We can pass a dummy array.
        goals_np = np.zeros(batch_size, dtype=np.float32)

        return [states_np, goals_np]
