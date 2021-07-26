'''
This version has a network trainer and workers, but for demonstration,
here the networks have all been removed and random designs are picked. 

'''



import torch
from replay_buffer_tensors import replay_buffer
import os
import logging
import numpy as np
import time
from simulation_runner import simulation_runner, terrain_grid_shape, reward_function, control_file
from simulation_runner import MAX_BLOCK_HEIGHT_HIGH,MIN_BLOCK_DISTANCE_LOW,MIN_BLOCK_DISTANCE_HIGH,MAX_BLOCK_HEIGHT_LOW
import traceback
from design_assembler import module_types, num_module_types, module_penalties
from design_assembler import add_module, module_vector_list_to_robot_name

# utility that helps manage the logging mess for multiple workers
# I downloaded the main file from
# https://github.com/jruere/multiprocessing-logging
import multiprocessing_logging
multiprocessing_logging.install_mp_handler()



cwd = os.path.dirname(os.path.realpath(__file__))

log_path =  os.path.join(cwd, "output.log")
logging.basicConfig(level=logging.INFO,
                    format='%(message)s', 
                    filename=log_path,
                    filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
print = logging.info # replace print with logging so that I don't have to find replace it


### hyperparameters
cpu_count = torch.multiprocessing.cpu_count()
# if cpu_count > 20:
#     NUM_SIM_WORKERS = 15
# elif cpu_count > 10:
#     NUM_SIM_WORKERS = 7
# elif cpu_count > 5:
#     NUM_SIM_WORKERS = 4
# else:
#     NUM_SIM_WORKERS = 2

NUM_SIM_WORKERS = 1 # for testing

N_ACTIONS = num_module_types
MAX_N_MODULES = 3
REPLAY_MEMORY_SIZE = 5000
NUM_ENVS = 3 # number of environments to run in each worker.
# ^ They will all have the same robot and terrain, and the policy will be called in batches.
SIM_TIME_STEPS = 250 # determines the farthest possible travel distance
NUM_EPISODES = 10000


def run_episode(policy_net,
            pipe,
            is_training_episode, # some episodes are for validation and don't get sent to buffer
            terrains, 
            sim_runner=None, 
            current_boltzmann_temp= 1, 
            n_designs = 1,
            print_str = ''):
    
    # initialize empty robot batch
    designs = torch.zeros(n_designs, N_ACTIONS*MAX_N_MODULES + MAX_N_MODULES)
    designs[:, N_ACTIONS*MAX_N_MODULES] = 1 # indicate which port adding now
    
    # loop dqn until done:
    for i_dqn in range(MAX_N_MODULES):

        # # run through generator to get actions
        # if is_training_episode:
        #     actions, actions_softmax = select_boltzmann_action(policy_net,designs,
        #                      terrains, current_boltzmann_temp)
        #     # actions, actions_softmax = select_epsgreedy_action(designs.to(device),
        #                      # terrains.to(device), 0.9)
        # else:
        #     actions, state_action_values = select_max_action(policy_net,designs,
        #                      terrains)

        # Random module added
        actions = torch.randint(low=0, high=N_ACTIONS, size=(n_designs,))
        # action_inds = torch.randint(0, N_ACTIONS, (designs.shape[0],))

        # add a module
        next_designs = torch.zeros_like(designs)
        for i_env in range(n_designs):
            next_designs[i_env,:], penalty = add_module(
                                    designs[i_env,:], 
                                    i_dqn, MAX_N_MODULES,
                                    actions[i_env])

        reward = -torch.tensor(penalty, dtype=torch.float32) # adding a module has no cost for now
        # But this addes the option to have a penalty for certain module types

        if i_dqn==(MAX_N_MODULES-1): # we are done
            non_final = torch.tensor(0, dtype=torch.bool)

            # convert one-hot into list module names
            for i_env in range(n_designs):
                # print(next_designs[i_env])
                mv = next_designs[i_env,:N_ACTIONS*MAX_N_MODULES].reshape(MAX_N_MODULES,N_ACTIONS)
                # print(mv)
                robot_name = module_vector_list_to_robot_name(mv)

            if is_training_episode:
                # run policy
                # robot_names_list = ['lll']

                # Here is where the simulation is reset and run
                sim_runner.load_robots(robot_name)
                rewards, power = sim_runner.run_sims(n_time_steps=SIM_TIME_STEPS)
                # if (sim_runner.reward_function == 'Testing Proxy' or 
                #     sim_runner.reward_function == 'Recorded Simulation'):
                #     time.sleep(0.1)

                reward += rewards.mean()
                # print(rewards.numpy())
                r_var = rewards.var()
                # print(r_var.numpy())
                if sim_runner.is_valid:
                    # print(terrains)
                    terrain_max = terrains.max().numpy()
                    print(print_str + ' simulated ' + str(robot_name) +
                        ' rewards ' +
                            np.array2string(rewards.numpy(),precision=1) 
                            + ' Terrain max ' + np.array2string(terrain_max,precision=3))

            if is_training_episode:
                # add to replay buffer
                for i_env in range(n_designs):
                    action = actions[i_env].numpy()
                    des_i = designs[i_env].numpy()
                    terr_i = terrains[i_env].numpy()
                    next_des_i = next_designs[i_env].numpy()
                    reward = reward.squeeze().numpy()
                    non_final_i = non_final.numpy()
                    r_var = r_var.numpy()
                    # print(non_final_i)
                    # print(des_i)
                    # print(next_des_i)
                    # print(reward)
                    # try:
                    # send only when reward > 0
                    if reward >= 0:
                        pipe.send([des_i, terr_i, action,
                                   next_des_i, reward, non_final_i, r_var])


            else:
                print(robot_name)

        else:
            non_final = torch.tensor(1, dtype=torch.bool)
        



        # else:
        #     # print('designs')
        #     # print(str(designs.cpu().numpy()))
        #     print('state_action_values')
        #     print(str(state_action_values.cpu().numpy()))
        #     print('Actions ' + str(actions.cpu().numpy()))
        #     # print('next_designs')
        #     # print(str(next_designs.cpu().numpy()))
        
        # hold designs for next step
        designs = next_designs


#   Runs simulations and pushes the results to memory buffer
def pusher_worker(policy_net, pipe,
         current_episode, NUM_EPISODES, worker_num):



    print('started pusher_worker ' + str(worker_num))

    sim_runner = simulation_runner(NUM_ENVS)
    # sim_runner = simulation_runner(NUM_ENVS, show_GUI= True)


    terrain_grid_shape = sim_runner.terrain_grid_shape
    while current_episode.value<NUM_EPISODES:
        with current_episode.get_lock():
            current_episode.value += 1
        i_episode = current_episode.value

        # Training episode:
        # select randomized terrain for training episode
        terrain = sim_runner.randomize_terrains()
        print_str_now = 'worker:' + str(worker_num) + ', ep:' + str(i_episode)
        # from matplotlib import pyplot as plt
        # plt.imshow(terrain[0][0].cpu().numpy())
        # print(terrain.shape)
        run_episode(policy_net, pipe, True,
             terrain, sim_runner,
              print_str = print_str_now )


        # # validation episode:
        # # test a selected few robots on a range of terrain
        # if (i_episode % VALIDATION_EP == 0 and i_episode>0):
        #     print('Boltzmann temp at ep ' + str(i_episode) + ': ' + str(current_boltzmann_temp))
        #     for terrain_block_height in np.linspace(
        #             sim_runner.MAX_BLOCK_HEIGHT_LOW,
        #             sim_runner.MAX_BLOCK_HEIGHT_HIGH, 3):
        #         terrain = sim_runner.randomize_terrains(
        #             terrain_block_height=terrain_block_height)

        #         terrain_max=torch.max(terrain).numpy().item()


        #         # compare with a range of real robots:
        #         test_robot_list = ['lll', 'lwl', 'wnw']
        #         test_robot_rewards = []
        #         out_str ='Test rewards: '
        #         for test_robot_name in test_robot_list:
        #             sim_runner.load_robots(test_robot_name)
        #             test_robot_rewards.append(sim_runner.run_sims())
        #             out_str += np.array2string(
        #                 test_robot_rewards[-1].numpy(),precision=1)
        #         print('--- eval at ep ' + str(i_episode) + ' ---')
        #         print('terrain max: ' + str(terrain_max))
        #         print('Test robots:' + str(test_robot_list))
        #         print( out_str )

        #         run_episode(policy_net,pipe,
        #             False,terrain) # for validation, don't simulate or store anything,
        #         # run with a range of terrains to check output

        #         print('-----------')

    pipe.send(None) # indicates the worker is done. Not sure if needed.
    print('finished pusher_worker ' + str(worker_num))
    return



if __name__== "__main__":
    collect_single_terr = True
    if collect_single_terr:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ### Initialize and load policy
        policy_net = torch.nn.Module()  # DUMMY POLICY
        # share memory for multiprocess
        policy_net.share_memory()

        state_size = [MAX_N_MODULES + N_ACTIONS * MAX_N_MODULES]
        terrain_size = terrain_grid_shape
        action_size = [1]

        r_list = np.asarray([
            [0, 1, 0, 0, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 1],
            [0, 0, 1, 0, 0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0, 0, 1],
            [0, 1, 0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 1, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 1]
        ])
        NUM_ENVS_TEST = 8
        # Just so that the terrain is the same

        sim_runner = simulation_runner(NUM_ENVS_TEST) #, show_GUI=True, gui_speed_factor =10
        np.random.seed(42)
        terrain = sim_runner.randomize_terrains(terrain_block_height=MAX_BLOCK_HEIGHT_HIGH)
        np.random.seed()
        reward_list = []
        r_var_list = []
        r_max_list = []
        r_name_list = []
        energy_list = []
        for i in range(r_list.shape[0]):
            # print(next_designs[i_env])
            mv = torch.tensor(r_list[i]).reshape(MAX_N_MODULES, N_ACTIONS)
            # print(mv)
            robot_name = module_vector_list_to_robot_name(mv)
            print(robot_name)
            # run policy
            # robot_names_list = ['lll']

            # Here is where the simulation is reset and run
            sim_runner.load_robots(robot_name)
            rewards, power = sim_runner.run_sims(n_time_steps=SIM_TIME_STEPS)
            # if (sim_runner.reward_function == 'Testing Proxy' or
            #     sim_runner.reward_function == 'Recorded Simulation'):
            #     time.sleep(0.1)
            print(power.numpy())
            reward = rewards.mean()
            # print(rewards.numpy())
            r_var = rewards.var()
            # print(r_var.numpy())
            r_var_list.append(r_var.numpy())
            reward_list.append(reward.numpy())
            r_max_list.append(rewards.max().numpy())
            r_name_list.append(robot_name)
            energy_list.append(power.mean().numpy() / reward_list[-1])
            if sim_runner.is_valid:
                # print(terrains)
                terrain_max = terrain.max().numpy()
                print(' simulated ' + str(robot_name) +
                      ' rewards ' +
                      np.array2string(rewards.numpy(), precision=1)
                      + ' Terrain max ' + np.array2string(terrain_max, precision=3)
                      + ' reward_variance ' + str(r_var) + 'reward_mean ' + str(reward)
                      + ' energy ' + str(energy_list[-1]))
        import matplotlib.pyplot as plt
        # print(r_max_list, r_var_list)
        plt.scatter(r_max_list, r_var_list)
        plt.xlabel("max_distance")
        plt.ylabel("distance_variance")
        plt.show()
        # plt.scatter(reward_list, r_var_list)
        # plt.show()
        plt.scatter(reward_list, energy_list)
        plt.xlabel("mean_distance")
        plt.ylabel("energy_cost_per_distance")
        plt.show()
        np.save("r_name", r_name_list)
        np.save("r_max", r_max_list)
        np.save("r_mean", reward_list)
        np.save("r_var", r_var_list)
        exit()

    # spawn processes
    torch.multiprocessing.set_start_method('spawn', force=True) # needed for CUDA drivers in parallel
    torch.multiprocessing.set_sharing_strategy('file_system')

    # device = torch.device('cpu')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    print('MAX_BLOCK_HEIGHT_LOW = ' + str(MAX_BLOCK_HEIGHT_LOW))
    print('MAX_BLOCK_HEIGHT_HIGH = ' + str(MAX_BLOCK_HEIGHT_HIGH))
    print('MIN_BLOCK_DISTANCE_LOW = ' + str(MIN_BLOCK_DISTANCE_LOW))
    print('MIN_BLOCK_DISTANCE_HIGH = ' + str(MIN_BLOCK_DISTANCE_HIGH))


    ### Initialize and load policy
    policy_net = torch.nn.Module() # DUMMY POLICY
    # share memory for multiprocess
    policy_net.share_memory()

    state_size = [MAX_N_MODULES + N_ACTIONS*MAX_N_MODULES]
    terrain_size = terrain_grid_shape
    action_size = [1] 

    ### Initialize replay buffer
    replay_memory = replay_buffer(REPLAY_MEMORY_SIZE, state_size, terrain_size, action_size)

    current_episode = torch.multiprocessing.Value('L', 0)

    processes = []
    pipes = []
    for worker_num in range(NUM_SIM_WORKERS): 
        parent_conn, child_conn = torch.multiprocessing.Pipe()
        pipes.append(parent_conn)
        p = torch.multiprocessing.Process(target=pusher_worker, 
                                args=(policy_net, child_conn,
                                    current_episode, 
                                    NUM_EPISODES, worker_num,))
        p.start()
        processes.append(p)
        time.sleep(0.01)

    num_p = len(pipes)

    # policy_net_copy is the policy network but on the gpu, 
    # it pushes to the cpu copy periodically.
    policy_net_copy =  torch.tensor(1) # DUMMY POLICY
    target_net =  torch.tensor(1) # DUMMY POLICY


    opt_ep = 0
    running = True
    alive_list= [True]*num_p # track which workers are alive
    while running:
        if current_episode.value >= NUM_EPISODES:
            running = False
        i_episode = current_episode.value

        # gather data from the pipes
        for i in range(num_p):
            process = processes[i]
            is_alive = process.is_alive()
            if not is_alive:
                alive_list[i] = False
            if is_alive:
                # Note: it seems to be better to pass numpy over the pipe
                # rather than sharing the entire buffer
                while pipes[i].poll():
                    # print('Pipe fileno: ' + str(pipes[i].fileno()))
                    pipe_read = pipes[i].recv()
                    if pipe_read is not None:
                        # if(pipe_read[4]<0):
                        #     continue
                        replay_memory.push(
                                torch.tensor(pipe_read[0]),
                                torch.tensor(pipe_read[1]), 
                                torch.tensor(pipe_read[2]), 
                                torch.tensor(pipe_read[3]),
                                torch.tensor(pipe_read[4]),
                                torch.tensor(pipe_read[5]),
                                torch.tensor(pipe_read[6]))

                        # save robot, terrain and reward every 100 datapoints
                        num_data = len(replay_memory)
                        print(f"replay buffer size: {num_data}")
                        if(num_data>0 and num_data % 30 == 0):
                            print(f"Saving data")
                            torch.save(replay_memory.memory_next_state, "designs.pt")
                            torch.save(replay_memory.memory_terrain, "terrains.pt")
                            torch.save(replay_memory.memory_reward, "rewards.pt")
                            torch.save(replay_memory.memory_rvar, "rvar.pt")
                            print(replay_memory.memory_terrain.shape)


                    else:
                        print("pipe return none")
                        # a None over the pipe indicates a redundant Done signal
                        alive_list[i] = False


        # if all the workers are done, end the loop
        if not(np.any(alive_list)):
            print('all workers ended')
            running = False

        ## This is where I'd put my network training... IF I HAD ONE!
        opt_ep += 1
        time.sleep(0.01) # keep loop from being too fast



    print('Done training')

