import torch
import numpy as np
import time


class replay_buffer(torch.nn.Module):
    def __init__(self, capacity, state_size, terrain_size, action_size):
        super(replay_buffer, self).__init__()
        self.capacity = torch.tensor(capacity, dtype=torch.long)
        self.memory_state = torch.zeros( [capacity]+ state_size, dtype=torch.float32)
        self.memory_terrain = torch.zeros( [capacity]+terrain_size, dtype=torch.float32)
        self.memory_action= torch.zeros( [capacity]+action_size, dtype=torch.float32)
        self.memory_next_state = torch.zeros( [capacity]+state_size, dtype=torch.float32)
        self.memory_reward= torch.zeros( [capacity]+[1], dtype=torch.float32)
        self.memory_non_final= torch.zeros( [capacity]+[1], dtype=torch.bool)
        self._size = torch.zeros(1, dtype=torch.long)
        self._next_idx = torch.zeros(1, dtype=torch.long)

    def can_sample(self,batch_size):
        return self._size>=batch_size

    def push(self, state, terrain, action, next_state, reward, non_final):
        """Saves a transition."""
        self.memory_state[self._next_idx,:] = state
        self.memory_terrain[self._next_idx,:,:] = terrain
        self.memory_action[self._next_idx,:] = action
        self.memory_next_state[self._next_idx,:] = next_state
        self.memory_reward[self._next_idx,:] = reward
        self.memory_non_final[self._next_idx,:] = non_final
        
        if self._next_idx >= self._size:
            self._size += 1
            # print(self._size)

        # self._next_idx = (self._next_idx + 1) % self.capacity
        self._next_idx = torch.remainder(self._next_idx + 1, self.capacity)

    def sample(self, batch_size):

        idxes = np.random.choice(self._size, batch_size, replace=False)
        states = self.memory_state[idxes,:]
        terrain = self.memory_terrain[idxes,:,:]
        action = self.memory_action[idxes,:]
        next_state = self.memory_next_state[idxes,:]
        reward = self.memory_reward[idxes,:]
        non_final = self.memory_non_final[idxes,:]

        return states, terrains, actions,next_states, rewards, non_finals


    def __len__(self):
        return self._size 


def sampler_worker(replay_memory):
    time_init = time.time()
    dt = 0
    print('started sampler_worker')
    while dt < 60:  
        if replay_memory.can_sample(5):
            states, terrains, actions, next_states, rewards, non_finals = replay_memory.sample(5)
            print('-----')
            print(str(states) + str(terrains[0,0]))
        else:
            print('replay_memory len: ' + str(len(replay_memory)))
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
            torch.tensor(worker_num),            
            torch.ones(1,3,3)*(time_now-time_init),z,z,z,z)
        idx_now  = replay_memory._next_idx
        if np.mod(idx_now,10)==0:
            print('worker ' + str(worker_num) + ' at idx '+ str(idx_now))
        dt = time_now - time_init
        # time.sleep(0.01)
        time.sleep(np.random.rand()*0.1)

if __name__== "__main__":

        # spawn processes
    torch.multiprocessing.set_start_method('spawn') # needed for CUDA drivers in parallel
    capacity = 200
    state_size = [1]
    terrain_size = [3,3]
    action_size = [1]

    replay_memory = replay_buffer(capacity, state_size,terrain_size,action_size)
    replay_memory.share_memory()

    processes = []
    for worker_num in range(3): 
        p = torch.multiprocessing.Process(target=pusher_worker, 
                                args=(replay_memory, worker_num,))
        p.start()
        processes.append(p)
        time.sleep(0.5)

    p = torch.multiprocessing.Process(target=sampler_worker, 
                           args=( replay_memory,))
    p.start()
    processes.append(p)
    time.sleep(0.01)

    # join processes. The main purpose of join() is to ensure that a child process has completed before the main process does anything that depends on the work of the child process.
    for worker_num in range(len(processes)):
        p  = processes[worker_num]
        p.join()
