import random
from collections import deque


class ReplayBuffer:
    def __init__(self, max_size: int):
        self.buffer = deque(maxlen=max_size)

    def store_state(self, experience: tuple):
        self.buffer.append(experience)

    def sample_buffer(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)
