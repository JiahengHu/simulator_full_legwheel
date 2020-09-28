

import torch
import random
from collections import namedtuple
import random
import numpy as np
import time

## Replay Memory
# We use experience replay memory for training the DQN.
class replay_buffer(object):

    def __init__(self, capacity, manager):
        self.capacity = capacity
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




def sampler_worker(replay_memory):
    time_init = time.time()
    dt = 0
    print('started sampler_worker')
    while dt < 60:  
        if replay_memory.can_sample(5):
            states, terrains, actions, next_states, rewards, non_finals = replay_memory.sample(5)
            print('-----')
            print(states)
            print(terrains)
        time.sleep(0.1)
        dt = time.time()- time_init

def pusher_worker(replay_memory, worker_num):

    print('started pusher_worker ' + str(worker_num))

    time_init = time.time()
    dt = 0
    z = torch.tensor(0.0)
    while dt < 60:  
        time_now = time.time()
        # 'state', 'terrain', 'action', 'next_state', 'reward', 'non_final'
        replay_memory.push(torch.tensor(time_now-time_init), torch.tensor(worker_num),z,z,z,z)
        dt = time_now - time_init
        time.sleep(0.01)

if __name__== "__main__":

        # spawn processes
    torch.multiprocessing.set_start_method('spawn') # needed for CUDA drivers in parallel

    manager = torch.multiprocessing.Manager()
    replay_memory = replay_buffer(100, manager)



    processes = []
    for worker_num in range(3): 
        p = torch.multiprocessing.Process(target=pusher_worker, 
                                args=(replay_memory, worker_num,))
        p.start()
        processes.append(p)
        time.sleep(0.01)

    p = torch.multiprocessing.Process(target=sampler_worker, 
                           args=( replay_memory,))
    p.start()
    processes.append(p)
    time.sleep(0.01)

    # join processes. The main purpose of join() is to ensure that a child process has completed before the main process does anything that depends on the work of the child process.
    for worker_num in range(len(processes)):
        p  = processes[worker_num]
        p.join()
        time.sleep(0.01)