import pickle
import numpy as np
import tinygrad.nn as nn
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
from tinygrad.nn.state import (
    get_state_dict,
    load_state_dict,
    get_parameters,
)


class DeepQNet:
    def forward(self, state):
        raise NotImplementedError("Forward not implemented")

    def get_weights(self):
        raise NotImplementedError("get_weights not implemented")

    def save(self, path):
        state_dict = get_state_dict(self)
        with open(path, "wb") as f:
            pickle.dump(state_dict, f)

    def load(self, path):
        state_dict = pickle.load(open(path, "rb"))
        state_dict = {k: v.detach() for k, v in state_dict.items()}
        load_state_dict(self, state_dict)

    def __call__(self, state):
        return self.forward(state)

    def clone(self) -> "DeepQNet":
        new_net = self.__class__()
        new_net.hard_update(self)
        return new_net

    def soft_update(self, other: "DeepQNet", tau: float):
        eval_state_dict = get_state_dict(other)
        target_state_dict = get_state_dict(self)

        for key in eval_state_dict:
            target_state_dict[key] = (
                tau * eval_state_dict[key].detach()
                + (1.0 - tau) * target_state_dict[key]
            )

        load_state_dict(self, target_state_dict)

    def hard_update(self, other: "DeepQNet"):
        other_params = get_parameters(other)
        self_params = get_parameters(self)
        for self_param, other_param in zip(self_params, other_params):
            self_param.assign(other_param.numpy().copy())


class SimpleCNN(DeepQNet):
    def __init__(self, input_dim):

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.flatten_size = 64 * input_dim[0] * input_dim[1]

        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x: np.ndarray):
        if x.ndim == 2:
            x = x[np.newaxis, np.newaxis, ...]
        elif x.ndim == 3:
            x = x[:, np.newaxis, ...]

        x = Tensor(x)
        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        x = self.conv3(x).relu()

        # Flatten the output for the fully connected layers
        x = x.reshape(shape=(-1, self.flatten_size))

        # Pass through fully connected layers
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        x = self.fc3(x)

        return x
