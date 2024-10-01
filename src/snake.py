import random
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from collections import deque
from gymnasium.spaces import Box, Discrete
from typing import Tuple, List, Dict, Any, Optional


class Snake(gym.Env):

    def __init__(
        self,
        snake_size: int = 3,
        grid_size: int = 11,
        max_moves_without_food: int = 100,
        reward_moves_towards_food: bool = True,
        seed: Optional[int] = None,
    ):
        self.snake_size = snake_size
        self.grid_size = grid_size
        self.max_moves_without_food = max_moves_without_food
        self.reward_moves_towards_food = reward_moves_towards_food
        self.max_score = (grid_size - 1) ** 2 - snake_size

        self.observation_space = Box(
            low=-1, high=4, shape=(self.grid_size, self.grid_size), dtype=np.float64
        )
        self.action_space = Discrete(4)
        self.grid_coordinates = set(
            (i, j) for i in range(1, grid_size - 1) for j in range(1, grid_size - 1)
        )
        self._action_to_direction = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
        self._action_to_string = {0: "LEFT", 1: "RIGHT", 2: "UP", 3: "DOWN"}

        if seed:
            self._set_seed(seed)

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def l1_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _create_snake(self) -> List[Tuple[int, int]]:
        """Create a snake with the initial size in the middle of the grid."""
        snake_head = (self.grid_size // 2, self.grid_size // 2)
        return deque(
            [(snake_head[0] + i, snake_head[1]) for i in range(self.snake_size)]
        )

    def _create_food(self) -> Tuple[int, int]:
        """Create a new food position in the empty grid coordinates."""
        return random.choice(list(self.grid_coordinates - set(self.snake)))

    def _move_snake(self, action: int):
        """Move the snake in the direction of the action."""
        direction = self._action_to_direction[action]
        self.snake.appendleft(
            (self.snake[0][0] + direction[0], self.snake[0][1] + direction[1])
        )
        if not self._check_food():
            self.snake.pop()

    def _check_collision(self) -> bool:
        """Check if the snake has collided with the wall or itself."""
        head = self.snake[0]
        has_hit_wall = head[0] in (0, self.grid_size - 1) or head[1] in (
            0,
            self.grid_size - 1,
        )
        has_hit_snake = len(self.snake) > len(set(self.snake))
        return has_hit_wall or has_hit_snake

    def _check_food(self) -> bool:
        """Check if the snake has eaten the food."""
        return self.food == self.snake[0]

    def _get_obs(self) -> np.ndarray:
        """Return the current observation of the environment."""

        # Initialize grid with border
        observation = np.full((self.grid_size, self.grid_size), -1, dtype=np.float64)
        observation[1:-1, 1:-1] = 0

        # Snake with distinguishable head & tail
        for i, j in self.snake:
            observation[i, j] = 1
        observation[self.snake[0][0], self.snake[0][1]] = 2
        observation[self.snake[-1][0], self.snake[-1][1]] = 1.5

        # Food
        observation[self.food[0], self.food[1]] = 4

        return observation

    def reset(self, *, seed=None, options=None):
        """Reset the environment to its initial state."""
        if seed:
            self._set_seed(seed)
        self.snake = self._create_snake()
        self.food = self._create_food()
        self.score = 0
        self.moves_without_food = 0
        self.steps_taken = 0
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment given an action."""
        reward = 0
        self.steps_taken += 1
        terminated = False
        truncated = False
        self._move_snake(action)

        if self._check_food():
            self.food = self._create_food()
            self.score += 1
            self.moves_without_food = 0
            reward = 1
        elif self._check_collision():
            terminated = True
            reward = -10
        else:
            self.moves_without_food += 1
            dist_to_food = self.l1_distance(self.snake[0], self.food)
            prev_dist_to_food = self.l1_distance(self.snake[1], self.food)
            reward = 0.02 if dist_to_food < prev_dist_to_food else -0.02
            truncated = self.moves_without_food >= self.max_moves_without_food

        return self._get_obs(), reward, terminated, truncated, {}

    @staticmethod
    def render_states(
        states: List[np.ndarray],
        filename: str,
        interval: int = 250,
    ) -> None:
        """Render a list of states."""
        fig = plt.figure()
        ims = []
        for state in states:
            im = plt.imshow(state, animated=True)
            plt.axis("off")
            ims.append([im])
        ani = animation.ArtistAnimation(
            fig, ims, interval=interval, blit=True, repeat_delay=1
        )
        ani.save(filename)
        plt.close()
