import numpy as np
from Replay import ReplayBuffer
from DeepQNet import DeepQNet, load_DeepQNet
from collections import deque


class DDQNAgent:
    def __init__(self, input_dim, n_actions, learning_rate, batch_size,
                 mem_size, gamma=0.95, replace=20, epsilon=1., epsilon_dec=0.999,
                 epsilon_min=0.01, TAU=0., model_paths=[]):
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.lr = learning_rate
        self.batch_size = batch_size
        self.step_counter = 0
        self.replace = replace
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.epsilon = epsilon
        self.memory = ReplayBuffer(mem_size)
        if not model_paths:
            print('Creating new model')
            self.q_eval = DeepQNet(input_dim, n_actions, learning_rate)
            self.q_target = DeepQNet(input_dim, n_actions, learning_rate)
        else:
            print('Loading model')
            self.q_eval = load_DeepQNet(model_paths[0])
            self.q_target = load_DeepQNet(model_paths[1])


    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def sample_memory(self):
        return self.memory.sample_buffer(self.batch_size, self.input_dim)

    def choose_action(self, cur_state):
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(range(self.n_actions))
        else:
            cur_state = np.array(cur_state).reshape((1,) + self.input_dim)
            actions = self.q_eval.predict(cur_state)
            action = np.argmax(actions)
        return action

    def replace_target_network(self):
        if not TAU:  # update target weights
            self.q_target.set_weights(self.q_eval.get_weights())
        else:  # soft update target weights
            q_network_theta = self.q_eval.get_weights()
            target_network_theta = self.q_target.get_weights()
            counter = 0
            for q_weight, target_weight in zip(q_network_theta, target_network_theta):
                target_weight = target_weight * (1-TAU) + q_weight * TAU
                target_network_theta[counter] = target_weight
                counter += 1
            self.q_target.set_weights(target_network_theta)

    def learn(self):
        states, actions, rewards, new_states, not_dones = self.sample_memory()

        q_pred = self.q_eval.predict(states)
        q_next = self.q_eval.predict(new_states)
        q_targ = self.q_target.predict(new_states)

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        max_actions = np.argmax(q_targ, axis=1).astype(int)

        q_targ[batch_index, actions] = rewards + self.gamma*q_targ[batch_index, max_actions]*not_dones
        _ = self.q_eval.fit(states, q_targ, epochs=1, verbose=0)

        self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon>self.epsilon_min else self.epsilon_min
        self.step_counter += 1

    def save_models(self, eval_name, targ_name):
        self.q_eval.save(eval_name)
        self.q_target.save(targ_name)
