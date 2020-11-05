

import torch
from replay_buffer import replay_buffer
# from design_assembler import module_types, num_module_types, module_penalties
# from design_assembler import add_module, module_vector_list_to_robot_name
# from dqn import dqn
import os
# import logging
import numpy as np
import time
from simulation_runner import simulation_runner, terrain_grid_shape
# import traceback



NUM_SIM_WORKERS = 1
REPLAY_MEMORY_SIZE = 5000
N_ACTIONS = 3
MAX_N_MODULES = 3
NUM_EPISODES = 20000
BATCH_SIZE = 100 # number of samples in a batch for dqn learning



def run_episode_test(replay_memory):
    


    terr_i = torch.zeros(terrain_grid_shape)
    des_i = torch.zeros(1, N_ACTIONS*MAX_N_MODULES + MAX_N_MODULES)
    next_des_i = torch.zeros(1, N_ACTIONS*MAX_N_MODULES + MAX_N_MODULES)
    action = torch.zeros(1, N_ACTIONS)
    reward = torch.zeros(1,1)
    non_final_i = torch.zeros(1,1)
    replay_memory.push(des_i, terr_i, action, 
                        next_des_i, reward, non_final_i)


#   Runs simulations and pushes the results to memory buffer
def pusher_worker(replay_memory,
         current_episode, max_episode, worker_num):

    print('started pusher_worker ' + str(worker_num))
    while current_episode.value<max_episode:
        with current_episode.get_lock():
            current_episode.value += 1
        i_episode = current_episode.value
        run_episode_test(replay_memory)
        time.sleep(0.01)


# samples memory and optimizes network
def sampler_worker(replay_memory, current_episode, max_episode):
    print('started sampler_worker')

    
    opt_ep = 0
    while current_episode.value < max_episode:
        i_episode = current_episode.value

        if len(replay_memory) >= BATCH_SIZE:

            # # Compute a mask of non-final states and concatenate the batch elements
            state_batch, terrain_batch, action_batch, next_state_batch, reward_batch, non_final_batch = replay_memory.sample(BATCH_SIZE)
            
        time.sleep(0.01) # keep loop from being too fast

if __name__== "__main__":

    # spawn processes
    torch.multiprocessing.set_start_method('spawn') # needed for CUDA drivers in parallel
    # torch.multiprocessing.set_sharing_strategy('file_system') # might be needed for opening and closing many files




    ### Initialize replay buffer
    manager = torch.multiprocessing.Manager()
    replay_memory = replay_buffer(REPLAY_MEMORY_SIZE, manager)
    current_episode = torch.multiprocessing.Value('L', 0)

    processes = []
    for worker_num in range(NUM_SIM_WORKERS): 
        p = torch.multiprocessing.Process(target=pusher_worker, 
                                args=(replay_memory,
                                    current_episode, 
                                    NUM_EPISODES, worker_num,))
        p.start()
        processes.append(p)
        time.sleep(0.01)


    ##Start sampler worker
    p = torch.multiprocessing.Process(target=sampler_worker, 
                           args=( replay_memory, 
                            current_episode, NUM_EPISODES,))
    p.start()
    processes.append(p)
    time.sleep(0.01)

    # # join processes. The main purpose of join() is to ensure that a child process has 
    # completed before the main process does anything that depends on the work of the child process.
    for worker_num in range(len(processes)):
        p  = processes[worker_num]
        p.join()
        time.sleep(0.01)


    # # watch for when all workers are dead, then end
    # running = True
    # while running:
    #     pushers_living = []
    #     for p in processes[:-1]:
    #         pushers_living.append(p.is_alive())
    #     sampler_living = processes[-1].is_alive()

    #     # end loop if the sampler dies, or all pushers die.
    #     if not(sampler_living) or not(np.any(pushers_living)):
    #         running = False
    #     time.sleep(5)

    # print('Done training')
    # # Done training

    # import test_selector
    # test_selector.run_test('', 'policy_net_params.pt')