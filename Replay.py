import numpy as np
from collections import deque
import random


class ReplayBuffer:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.memory = deque(maxlen=maxlen)

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done,))

    def sample_buffer(self, batch_size, state_shape):
        states, actions, rewards, new_states, dones = zip(*random.choices(self.memory,
                                                                          k=batch_size))
        states = np.array(states).reshape((batch_size,) + state_shape)
        actions = np.array(actions, dtype='int').reshape(batch_size)
        rewards = np.array(rewards).reshape(batch_size)
        new_states = np.array(new_states).reshape((batch_size,) + state_shape)
        not_dones = np.array(dones, dtype=np.bool).reshape(batch_size)
        return states, actions, rewards, new_states, not_dones

    def clear(self):
        self.memory = deque(maxlen=self.maxlen)

    def __len__(self):
        return len(self.memory)
