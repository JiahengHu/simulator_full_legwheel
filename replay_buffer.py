import torch
import numpy as np


## Replay Memory
# We use experience replay memory for training the DQN.
class replay_buffer(object):

    def __init__(self, capacity, manager=None):
        self.capacity = capacity
        if manager is None:
            self.memory = list()
        else:
            self.memory = manager.list()
        self._next_idx = 0

    def can_sample(self,batch_size):
        return len(self.memory)>=batch_size

    def push(self, state, terrain, action, next_state, reward, non_final):
        """Saves a transition."""
        data = (state, terrain, action, next_state, reward, non_final)
        if self._next_idx >= len(self.memory):
            self.memory.append(data)
        else:
            self.memory[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self.capacity

    def sample(self, batch_size):

        idxes = np.random.choice(len(self.memory), batch_size, replace=False)
        states, terrains, actions, next_states, rewards, non_finals = [], [], [], [], [], []
        for i in idxes:
            state, terrain, action, next_state, reward, non_final = self.memory[i]
            states.append(state)
            terrains.append(terrain)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            non_finals.append(non_final)

        return torch.stack(states), torch.stack(terrains), \
                torch.stack(actions), torch.stack(next_states), \
                torch.stack(rewards), torch.stack(non_finals)
            
    def __len__(self):
        return len(self.memory)


# from collections import namedtuple
# import random

# ## Replay Memory
# # We use experience replay memory for training the DQN.
# Transition = namedtuple('Transition',
#                         ('state', 'terrain', 'action', 'next_state', 'reward', 'non_final'))
# class replay_buffer(object):

#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.memory = []
#         self.position = 0

#     def push(self, *args):
#         """Saves a transition."""
#         if len(self.memory) < self.capacity:
#             self.memory.append(None)
#         self.memory[self.position] = Transition(*args)
#         self.position = (self.position + 1) % self.capacity

#     def sample(self, batch_size):
#         transitions= random.sample(self.memory, batch_size)
#         batch = Transition(*zip(*transitions))
#         return batch
#     # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
#     # detailed explanation). This converts batch-array of Transitions
#     # to Transition of batch-arrays.
            
#     def __len__(self):
#         return len(self.memory)

