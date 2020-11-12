'''
This seems to work


'''

import torch
import numpy as np
import time

STATE_SIZE = [10]
TERRAIN_SIZE = [50,100]

class replay_buffer(object):
    def __init__(self, capacity, state_size, terrain_size, action_size):
        self.capacity = capacity
        self.memory_state = torch.zeros( [capacity]+ state_size, dtype=torch.float32)
        self.memory_terrain = torch.zeros( [capacity]+terrain_size, dtype=torch.float32)
        self.memory_action= torch.zeros( [capacity]+action_size, dtype=torch.float32)
        self.memory_next_state = torch.zeros( [capacity]+state_size, dtype=torch.float32)
        self.memory_reward= torch.zeros( [capacity]+[1], dtype=torch.float32)
        self.memory_non_final= torch.zeros( [capacity]+[1], dtype=torch.bool)
        self._size = 0
        self._next_idx = 0

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

        self._next_idx = (self._next_idx + 1) % self.capacity
        # self._next_idx = torch.remainder(self._next_idx + 1, self.capacity)

    def sample(self, batch_size):

        idxes = np.random.choice(self._size, batch_size, replace=False)
        states = self.memory_state[idxes,:]
        terrains = self.memory_terrain[idxes,:,:]
        actions = self.memory_action[idxes,:]
        next_states = self.memory_next_state[idxes,:]
        rewards = self.memory_reward[idxes,:]
        non_finals = self.memory_non_final[idxes,:]

        return states, terrains, actions,next_states, rewards, non_finals


    def __len__(self):
        return self._size 


# def sampler_worker(replay_memory):
def sample_now(replay_memory):
    if replay_memory.can_sample(5):
        states, terrains, actions, next_states, rewards, non_finals = replay_memory.sample(5)
        print('-----')
        print(str(states[:,0]) + str(terrains[:,0,0]))
    else:
        print('replay_memory len: ' + str(len(replay_memory)))

def pusher_worker(worker_num, pipe):


    print('started pusher_worker ' + str(worker_num))
    time_init = time.time()
    dt = 0
    z = torch.tensor(0.0)
    while dt < 30:  
        time_now = time.time()
        data = (torch.ones([1] + STATE_SIZE)*worker_num,
                torch.ones([1] + TERRAIN_SIZE)*(time_now-time_init), 
                 z,z,z,z)
        pipe.send( data )
        dt = time_now - time_init
        # time.sleep(0.01)
        time.sleep(np.random.rand()*0.75)

if __name__== "__main__":

        # spawn processes
    torch.multiprocessing.set_start_method('spawn') # needed for CUDA drivers in parallel
    capacity = 200
    state_size =STATE_SIZE# [1]
    terrain_size = TERRAIN_SIZE#[10,10]
    action_size = [1]

    replay_memory = replay_buffer(capacity, state_size,terrain_size,action_size)

    processes = []
    pipes = []
    for worker_num in range(3): 
        parent_conn, child_conn = torch.multiprocessing.Pipe()
        pipes.append(parent_conn)
        p = torch.multiprocessing.Process(target=pusher_worker, 
                                args=(worker_num,child_conn,))
        p.start()
        processes.append(p)
        time.sleep(0.1)

    num_p = len(pipes)
    while True:
        alive_list= []
        for i in range(num_p):
            pipe = pipes[i]
            process = processes[i]
            i_alive = process.is_alive()
            alive_list.append(i_alive)
            if i_alive:
                while pipe.poll():
                    pipe_read = pipe.recv()
                    replay_memory.push(pipe_read[0], pipe_read[1], 
                        pipe_read[2], pipe_read[3], pipe_read[4], pipe_read[5] )
                    print('pushed to memory ' + str(pipe_read[0][0,0].item()) + 
                        ' ' +str( pipe_read[1][0,0,0].item())) 
        sample_now(replay_memory)
        print('Mem len: ' + str(len(replay_memory)))
        time.sleep(0.5)
        if not(np.any(alive_list)):
            break




    # join processes. The main purpose of join() is to ensure that a child process has completed before the main process does anything that depends on the work of the child process.
    for worker_num in range(len(processes)):
        p  = processes[worker_num]
        p.join()
