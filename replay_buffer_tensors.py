import torch
import numpy as np


## Replay Memory
# We use experience replay memory for training the DQN.
class replay_buffer(object):
    def __init__(self, capacity, state_size, terrain_size, action_size):
        self.capacity = capacity
        self.memory_state = torch.zeros( [capacity]+ state_size, dtype=torch.float32)
        self.memory_terrain = torch.zeros( [capacity]+terrain_size, dtype=torch.float32)
        self.memory_action= torch.zeros( [capacity]+action_size, dtype=torch.long)
        self.memory_next_state = torch.zeros( [capacity]+state_size, dtype=torch.float32)
        self.memory_reward= torch.zeros( [capacity], dtype=torch.float32)
        self.memory_non_final= torch.zeros( [capacity], dtype=torch.bool)
        self.memory_rvar = torch.zeros([capacity], dtype=torch.float32)
        self._size = 0
        self._next_idx = 0

    def can_sample(self,batch_size):
        return self._size>=batch_size

    def push(self, state, terrain, action, next_state, reward, non_final, rvar):
        """Saves a transition."""
        self.memory_state[self._next_idx,:] = state
        self.memory_terrain[self._next_idx,:,:] = terrain
        self.memory_action[self._next_idx,:] = action
        self.memory_next_state[self._next_idx,:] = next_state
        self.memory_reward[self._next_idx] = reward
        self.memory_non_final[self._next_idx] = non_final
        self.memory_rvar[self._next_idx] = rvar

        self._next_idx = (self._next_idx + 1) % self.capacity

        if self._next_idx >= self._size:
            self._size += 1
            # print(self._size)
        # self._next_idx = torch.remainder(self._next_idx + 1, self.capacity)

    def sample(self, batch_size):

        idxes = np.random.choice(self._size, batch_size, replace=False)
        states = self.memory_state[idxes,:]
        terrains = self.memory_terrain[idxes,:,:]
        actions = self.memory_action[idxes,:]
        next_states = self.memory_next_state[idxes,:]
        rewards = self.memory_reward[idxes]
        non_finals = self.memory_non_final[idxes]
        rvar = self.memory_rvar[idxes]
        return states, terrains, actions,next_states, rewards, non_finals, rvar


    def __len__(self):
        return self._size 

    def get_dict(self):
        # todo: only save positive rewards?
        dict_out = dict()
        dict_out['capacity'] = self.capacity
        dict_out['states']   = self.memory_state 
        dict_out['terrains']   = self.memory_terrain 
        dict_out['actions'] = self.memory_action
        dict_out['next_states']   = self.memory_next_state 
        dict_out['rewards']  = self.memory_reward
        dict_out['non_finals']  = self.memory_non_final
        dict_out['rvar'] = self.memory_rvar
        dict_out['size'] = self._size
        return dict_out