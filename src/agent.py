import numpy as np
import gymnasium as gym
import tinygrad.nn as nn

from tinygrad import Tensor
from typing import List
from .networks import DeepQNet
from dataclasses import dataclass
from .buffers import ReplayBuffer


class DDQNAgent:
    def __init__(
        self,
        q_eval: DeepQNet,
        q_target: DeepQNet,
        n_actions: int = 4,
        learning_rate: float = 0.00025,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_dec: float = 0.999,
        epsilon_min: float = 0.01,
        tau: float = 0.005,
        max_buffer_size: int = 10_000,
        lr: float = 3e-4,
        update_target_every: int = 500,
        seed=None,
    ):
        self.q_eval = q_eval
        self.q_target = q_target
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.tau = tau
        self.update_target_every = update_target_every
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(max_buffer_size)

        self.optimizer = nn.optim.Adam(
            nn.state.get_parameters(self.q_eval),
            lr=lr,
        )
        if seed:
            np.random.seed(seed)

    def choose_action(self, state: np.ndarray) -> int:
        """Choose an action based on the epsilon-greedy policy."""
        if np.random.uniform() < self.epsilon:
            return np.random.choice(range(self.n_actions))
        else:
            return self.q_eval.forward(state).numpy().argmax().item()

    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> None:
        """Store the experience in the replay buffer."""
        self.memory.store_state((state, action, reward, next_state, terminated))

    def sample_memory(self, batch_size: int) -> List[np.ndarray]:
        """Sample a batch of experiences from the replay buffer."""
        buffer_sample = self.memory.sample_buffer(batch_size)
        states = np.array([exp[0] for exp in buffer_sample], dtype=np.float32)
        actions = np.array([exp[1] for exp in buffer_sample], dtype=np.int32)
        rewards = np.array([exp[2] for exp in buffer_sample], dtype=np.float32)
        next_states = np.array([exp[3] for exp in buffer_sample], dtype=np.float32)
        terminated = np.array([exp[4] for exp in buffer_sample], dtype=np.float32)

        return states, actions, rewards, next_states, terminated

    def learn(self, batch_size: int = 32) -> float:
        """Update the Q-network parameters using the DQN algorithm."""
        if len(self.memory) < batch_size:
            return
        self.learn_step_counter += 1

        states, actions, rewards, next_states, terminated = self.sample_memory(
            batch_size
        )

        Tensor.no_grad = True
        q_predictions = self.q_eval.forward(states).numpy()
        q_targets = self.q_target.forward(next_states).numpy()
        next_actions = np.argmax(q_targets, axis=1)

        batch_index = np.arange(batch_size, dtype=np.int32)
        q_predictions[batch_index, actions] = rewards + self.gamma * q_targets[
            batch_index, next_actions
        ] * (1 - terminated)
        Tensor.no_grad = False

        Tensor.training = True
        loss = self.q_eval.forward(states).sub(Tensor(q_predictions)).pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        Tensor.training = False

        self.epsilon = max(self.epsilon * self.epsilon_dec, self.epsilon_min)

        if self.learn_step_counter % self.update_target_every == 0:
            self.q_target.hard_update(self.q_eval)
        else:
            self.q_target.soft_update(self.q_eval, self.tau)

        return loss.numpy()
