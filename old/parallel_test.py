'''
An error that shows up after a few seconds:

Process Process-3:
Traceback (most recent call last):
  File "/home/cobracommander/anaconda3/lib/python3.7/multiprocessing/process.py", line 297, in _bootstrap
    self.run()
  File "/home/cobracommander/anaconda3/lib/python3.7/multiprocessing/process.py", line 99, in run
    self._target(*self._args, **self._kwargs)
  File "/home/cobracommander/mobile_design_RL/parallel_test.py", line 80, in pusher_worker
    torch.tensor(worker_num),z,z,z,z)
  File "/home/cobracommander/mobile_design_RL/parallel_test.py", line 26, in push
    self.memory.append(data)
  File "<string>", line 2, in append
  File "/home/cobracommander/anaconda3/lib/python3.7/multiprocessing/managers.py", line 834, in _callmethod
    raise convert_to_error(kind, result)
multiprocessing.managers.RemoteError: 
---------------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/cobracommander/anaconda3/lib/python3.7/multiprocessing/managers.py", line 234, in serve_client
    request = recv()
  File "/home/cobracommander/anaconda3/lib/python3.7/multiprocessing/connection.py", line 251, in recv
    return _ForkingPickler.loads(buf.getbuffer())
  File "/home/cobracommander/anaconda3/lib/python3.7/site-packages/torch/multiprocessing/reductions.py", line 276, in rebuild_storage_fd
    fd = df.detach()
  File "/home/cobracommander/anaconda3/lib/python3.7/multiprocessing/resource_sharer.py", line 58, in detach
    return reduction.recv_handle(conn)
  File "/home/cobracommander/anaconda3/lib/python3.7/multiprocessing/reduction.py", line 185, in recv_handle
    return recvfds(s, 1)[0]
  File "/home/cobracommander/anaconda3/lib/python3.7/multiprocessing/reduction.py", line 161, in recvfds
    len(ancdata))
RuntimeError: received 0 items of ancdata
---------------------------------------------------------------------------


'''

import torch
import random
from collections import namedtuple
import random
import numpy as np
import time


## Replay Memory
# We use experience replay memory for training the DQN.
class replay_buffer(object):

    def __init__(self, capacity, managed_list):
        self.capacity = capacity
        self.memory = managed_list
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
        # print(len(replay_memory))
        replay_memory.push(
            # torch.tensor(time_now-time_init), 
            torch.ones(1,10)*(time_now-time_init), 
            torch.tensor(worker_num),z,z,z,z)
        dt = time_now - time_init
        time.sleep(0.01)

if __name__== "__main__":

        # spawn processes
    torch.multiprocessing.set_start_method('spawn') # needed for CUDA drivers in parallel

    manager = torch.multiprocessing.Manager()
    buffer_mem  = manager.list()
    replay_memory = replay_buffer(1000, buffer_mem)



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