'''
TODO 
- save file with all recorded data so far, so that later I can 
play it back faster and see if I can improve training speed
- during eval print out not just the one that just evaled,
but all possible state action values on the tree, recursively called,
( can be used to get the probability of random selection of each?)

'''



import torch
from replay_buffer import replay_buffer
from design_assembler import module_types, num_module_types
from design_assembler import add_module, module_vector_list_to_robot_name
from simulation_runner import simulation_runner, terrain_grid_shape
from dqn import dqn
import torch.nn.functional as F
import random
import os
import logging
import numpy as np
import time


cwd = os.path.dirname(os.path.realpath(__file__))

# log_path =  os.path.join(cwd, "output.log")
# logging.basicConfig(level=logging.INFO,
#                     format='%(message)s', 
#                     filename=log_path,
#                     filemode='w')
# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# formatter = logging.Formatter('%(message)s')
# console.setFormatter(formatter)
# logging.getLogger('').addHandler(console)
# print = logging.info


### hyperparameters
cpu_count = torch.multiprocessing.cpu_count()
if cpu_count > 6:
    NUM_SIM_WORKERS = 6
else:
    NUM_SIM_WORKERS = 1

RECORD_REWARDS = False # flag to keep some statistics on the rewards obtained by simulation
REPLAY_MEMORY_SIZE = 5000
LR_INIT = 1e-4
N_ACTIONS = num_module_types
MAX_N_MODULES = 3
NUM_ENVS = 3 # number of environments to run in parallel
SIM_TIME_STEPS = 100
NUM_EPISODES = 20000
TARGET_UPDATE = 100 # how many episodes to delay target net from policy net
BATCH_SIZE = 100 # number of samples in a batch for dqn learning
BOLTZMANN_TEMP_START = 10 
BOLTZMANN_TEMP_MIN = 1
BOLTZMANN_TEMP_DECAY_CONST = 1./4000 # T = T0*exp(-c*episode) e.g. 10*np.exp(-np.array([0, 1000, 5000])/1000)
# RELOAD_WEIGHTS=True # if true, looks for policy_net_weights.pt to load in
RELOAD_WEIGHTS=False # if true, looks for policy_net_weights.pt to load in
SAVE_EP = 100 # how many episodes to wait between saving
VALIDATION_EP = 100 # how many episdoes to wait between validations


# For testing only
# BATCH_SIZE = 10 # number of samples in a batch for dqn learning
# VALIDATION_EP = 10 # how many episdoes to wait between validations



#   Runs simulations and pushes the results to memory buffer
def pusher_worker(policy_net,
        replay_memory,
         current_episode, max_episode, worker_num):
    

    # temperature for boltzmann exploration 
    def boltzmann_temp(episode):
        return max(BOLTZMANN_TEMP_START*np.exp(-episode*BOLTZMANN_TEMP_DECAY_CONST),
                    BOLTZMANN_TEMP_MIN)

    # select an action randomly with boltzmann exploration
    def select_boltzmann_action(policy_net,designs, terrains, boltzmann_temp=1):
        with torch.no_grad():
            actions_out = policy_net(designs, terrains)
            # Scale by a temperature factor. as temperature gets lower, tends towards uniform outputs. as higher, tends towards true max. 
            actions_softmax = F.softmax(actions_out/boltzmann_temp, dim=-1)
        
        # TODO: replace with torch categorical 
        action_inds = []
        for ind in range(actions_softmax.shape[0]):
            chosen = random.uniform(0, 1)
            cumulative = 0
            for action_index in range(N_ACTIONS):
                cumulative += actions_softmax[0][action_index].item() # [0] bc first dimension is blank for stacking
                if cumulative >= chosen:
                    action_inds.append(action_index)
                    break 
        return torch.tensor(action_inds, dtype=torch.long), actions_softmax

    def select_epsgreedy_action(policy_net,designs, terrains, eps):
        with torch.no_grad():
            actions_out = policy_net(designs, terrains)
        if np.random.rand()>eps:
            action_inds = actions_out.max(1)[1].cpu()
        else:
            action_inds = torch.randint(0, N_ACTIONS, (designs.shape[0],))
        return action_inds, actions_out 

    # selects an action with the max Q value from all actions.
    def select_max_action(policy_net,designs, terrains):
        with torch.no_grad():
            actions_out = policy_net(designs, terrains)
            action_inds = actions_out.max(1)[1]
            return action_inds, actions_out 



    def run_episode(policy_net,replay_memory,is_training_episode,
                terrains, 
                sim_runner_now=None, 
                current_boltzmann_temp= 1, print_str = ''):
        
        # initialize empty robot batch
        designs = torch.zeros(1, N_ACTIONS*MAX_N_MODULES + MAX_N_MODULES)
        designs[:, N_ACTIONS*MAX_N_MODULES] = 1 # indicate which port adding now
        
        # loop dqn until done:
        for i_dqn in range(MAX_N_MODULES):

            # run through generator to get actions
            if is_training_episode:
                actions, actions_softmax = select_boltzmann_action(policy_net,designs,
                                 terrains, current_boltzmann_temp)
                # actions, actions_softmax = select_epsgreedy_action(designs.to(device),
                                 # terrains.to(device), 0.9)
            else:
                actions, state_action_values = select_max_action(policy_net,designs,
                                 terrains)

            # add a module
            next_designs = torch.zeros_like(designs)
            for i_env in range(1):
                next_designs[i_env,:] = add_module(
                                        designs[i_env,:], 
                                        i_dqn, MAX_N_MODULES,
                                        actions[i_env])

            reward = torch.zeros(1) # adding a module has no cost for now

            if i_dqn==(MAX_N_MODULES-1): # we are done
                non_final = torch.tensor(0, dtype=torch.bool)

                # convert one-hot into list module names
                for i_env in range(1):
                    # print(next_designs[i_env])
                    mv = next_designs[i_env,:N_ACTIONS*MAX_N_MODULES].reshape(MAX_N_MODULES,N_ACTIONS)
                    # print(mv)
                    robot_name = module_vector_list_to_robot_name(mv)

                if is_training_episode:
                    # run policy
                    # robot_names_list = ['lll']
                    sim_runner.load_robots(robot_name)
                    rewards = sim_runner.run_sims()
                    reward += rewards.mean()
                    if sim_runner.is_valid:
                        # print(terrains)
                        terrain_max = terrains.max().numpy()
                        print(print_str + ' simulated ' + str(robot_name) +
                            ' rewards ' +
                                np.array2string(rewards.numpy(),precision=1) 
                                + ' Terrain max ' + np.array2string(terrain_max,precision=3) 
)
                else:
                    print(robot_name)

            else:
                non_final = torch.tensor(1, dtype=torch.bool)
            

            if is_training_episode:
                # add to replay buffer
                for i_env in range(1):
                    action = actions[i_env].unsqueeze(0).clone()
                    # reward = reward.unsqueeze(0).clone()
                    reward = reward.squeeze().clone()
                    replay_memory.push(designs[i_env].clone(), terrains[i_env].clone(),
                        action, next_designs[i_env].clone(), reward, non_final.clone())
            else:
                # print('designs')
                # print(str(designs.cpu().numpy()))
                print('state_action_values')
                print(str(state_action_values.cpu().numpy()))
                print('Actions ' + str(actions.cpu().numpy()))
                # print('next_designs')
                # print(str(next_designs.cpu().numpy()))
            
            # hold designs for next step
            designs = next_designs




    print('started pusher_worker ' + str(worker_num))

    sim_runner = simulation_runner(NUM_ENVS)
    # sim_runner = simulation_runner(NUM_ENVS, show_GUI= True)
# 

    terrain_grid_shape = sim_runner.terrain_grid_shape
    while current_episode.value<max_episode:
        with current_episode.get_lock():
            current_episode.value += 1
        i_episode = current_episode.value


        # select randomized terrain for training episode
        terrain = sim_runner.randomize_terrains()
        current_boltzmann_temp = boltzmann_temp(i_episode)  # anneals temp
        print_str_now = 'worker:' + str(worker_num) + ', ep:' + str(i_episode)
        run_episode(policy_net,replay_memory,True,terrain, sim_runner,
              current_boltzmann_temp =  current_boltzmann_temp,
              print_str = print_str_now )


        # validation episode
        if (i_episode % VALIDATION_EP == 0 and i_episode>0):
            print('Boltzmann temp at ep ' + str(i_episode) + ': ' + str(current_boltzmann_temp))
            for terrain_block_height in np.linspace(
                    sim_runner.MAX_BLOCK_HEIGHT_LOW,
                    sim_runner.MAX_BLOCK_HEIGHT_HIGH, 3):
                terrain = sim_runner.randomize_terrains(
                    terrain_block_height=terrain_block_height)

                terrain_max=torch.max(terrain).numpy().item()


                # compare with a range of real robots:
                test_robot_list = ['lll', 'lwl', 'wnw']
                test_robot_rewards = []
                out_str ='Test rewards: '
                for test_robot_name in test_robot_list:
                    sim_runner.load_robots(test_robot_name)
                    test_robot_rewards.append(sim_runner.run_sims())
                    out_str += np.array2string(
                        test_robot_rewards[-1].numpy(),precision=1)
                print('--- eval at ep ' + str(i_episode) + ' ---')
                print('terrain max: ' + str(terrain_max))
                print('Test robots:' + str(test_robot_list))
                print( out_str )

                run_episode(policy_net,replay_memory,False,terrain) # for validation, don't simulate or store anything,
                # run with a range of terrains to check output

                print('-----------')


# samples memory and optimizes network
def sampler_worker(policy_net,replay_memory, device,current_episode, max_episode):
    print('started sampler_worker')

    # policy_net_copy = policy_net.to(device)


    policy_net_copy = dqn( 
        terrain_in_shape = policy_net.terrain_in_shape ,
        n_module_types= policy_net.n_module_types,
        max_num_modules=policy_net.max_num_modules,
        kernel_size=policy_net.kernel_size,
        n_channels= policy_net.n_channels,
        n_fc_layers=policy_net.n_fc_layers,
        env_vect_size=policy_net.env_vect_size,
        hidden_layer_size=policy_net.hidden_layer_size).to(device)

    target_net = dqn( 
        terrain_in_shape = policy_net.terrain_in_shape ,
        n_module_types= policy_net.n_module_types,
        max_num_modules=policy_net.max_num_modules,
        kernel_size=policy_net.kernel_size,
        n_channels= policy_net.n_channels,
        n_fc_layers=policy_net.n_fc_layers,
        env_vect_size=policy_net.env_vect_size,
        hidden_layer_size=policy_net.hidden_layer_size).to(device)


    optimizer = torch.optim.Adam(policy_net_copy.parameters(),
                   lr=LR_INIT, weight_decay= 1e-4)


    policy_net_copy.load_state_dict(policy_net.state_dict())
    target_net.load_state_dict(policy_net_copy.state_dict())
    target_net.eval()
    opt_ep = 0
    while current_episode.value < max_episode:

        i_episode = current_episode.value

        if len(replay_memory) >= BATCH_SIZE:

            # # Compute a mask of non-final states and concatenate the batch elements
            state_batch, terrain_batch, action_batch, next_state_batch, reward_batch, non_final_batch = replay_memory.sample(BATCH_SIZE)
            state_batch = state_batch.to(device)
            terrain_batch =terrain_batch.to(device)
            action_batch = action_batch.to(device)
            next_state_batch =next_state_batch.to(device)
            reward_batch =reward_batch.to(device)
            non_final_mask = non_final_batch
            non_final_next_states = next_state_batch[non_final_mask]
            non_final_terrains = terrain_batch[non_final_mask]

            # forward pass with autograd
            # Compute Q(s_t, a)
            state_action_values_raw = policy_net_copy(
                                     state_batch,terrain_batch)
            state_action_values = state_action_values_raw.gather(
                                      1, action_batch).squeeze()

            # Compute V(s_{t+1}) for next states.
            next_state_values = torch.zeros(BATCH_SIZE, device=device)
            if len(non_final_next_states)>0:
                with torch.no_grad():
                    next_state_values[non_final_mask] = target_net(
                        non_final_next_states, non_final_terrains).max(1)[0].detach()

            # print(next_state_values.shape)
            # print(reward_batch.shape)
            # print(state_action_values.shape)

            # Compute the expected Q values
            expected_state_action_values = next_state_values + reward_batch


            # Compute Huber loss
            # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
            # compute mean squared error loss
            loss = F.mse_loss(state_action_values,
                              expected_state_action_values)


            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            # clamp the gradients
            for param in policy_net_copy.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()

            # print('optimized at step ' + str(i_episode))

            opt_ep += 1


            if (opt_ep) % 10==0 and opt_ep>0:
                # make sure the cpu copy has the most recent weights
                policy_net.load_state_dict(policy_net_copy.state_dict())

            if (opt_ep) % 50==0 and opt_ep>0:
                print('Loss at opt_ep. ' + str(opt_ep) + ': ' + str(loss.detach().cpu().numpy()))


            # Update the target network, copying all weights and biases in DQN
            if (opt_ep) % TARGET_UPDATE == 0 and opt_ep>0:
                target_net.load_state_dict(policy_net_copy.state_dict())

            if (opt_ep % 10000)==0 and opt_ep>10000:
                for param_group in optimizer.param_groups:
                    # half the learning rate periodically
                    param_group['lr'] = param_group['lr']/2.
                    print( 'LR: ' + str(param_group['lr']) )


            if (opt_ep % SAVE_EP == 0):
                PATH = os.path.join(cwd, 'policy_net_params.pt')
                save_dict = dict()
                save_dict['policy_net_state_dict'] = policy_net.state_dict()
                save_dict['terrain_in_shape'] = policy_net.terrain_in_shape
                save_dict['n_module_types'] = policy_net.n_module_types
                save_dict['max_num_modules'] = policy_net.max_num_modules
                save_dict['kernel_size']= policy_net.kernel_size
                save_dict['n_channels']=policy_net.n_channels
                save_dict['n_fc_layers']=policy_net.n_fc_layers
                save_dict['env_vect_size']=policy_net.env_vect_size
                save_dict['hidden_layer_size']=policy_net.hidden_layer_size
                save_dict['i_episode'] = i_episode
                torch.save(save_dict, PATH)

        time.sleep(0.01) # keep loop from being too fast

if __name__== "__main__":

    # spawn processes
    torch.multiprocessing.set_start_method('spawn') # needed for CUDA drivers in parallel


    # device = torch.device('cpu')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    
    ### Initialize and load policy
    # initiate the policy network 
    PATH = os.path.join(cwd, 'policy_net_params.pt')

    if RELOAD_WEIGHTS and os.path.exists(PATH):
        save_dict = torch.load(PATH)
        PATH = os.path.join(cwd, 'policy_net_params.pt')
        save_dict = dict()

        policy_net = dqn( 
            terrain_in_shape = save_dict['terrain_in_shape'] ,
            n_module_types= save_dict['n_module_types'] ,
            max_num_modules=save_dict['max_num_modules'] ,
            kernel_size=save_dict['kernel_size'],
            n_channels= save_dict['n_channels'],
            n_fc_layers=save_dict['n_fc_layers'],
            env_vect_size=save_dict['env_vect_size'],
            hidden_layer_size=save_dict['hidden_layer_size'])
        policy_net.load_state_dict( save_dict['policy_net_state_dict'])
        print('Reloaded weights from ' + PATH)
    else:
        print('Creating ' + PATH)
        policy_net = dqn(terrain_grid_shape, 
                     max_num_modules = MAX_N_MODULES)

    # share memory for multiprocess
    policy_net.share_memory()


    ### Initialize replay buffer
    manager = torch.multiprocessing.Manager()
    replay_memory = replay_buffer(REPLAY_MEMORY_SIZE, manager)
    current_episode = torch.multiprocessing.Value('L', 0)

    processes = []
    for worker_num in range(NUM_SIM_WORKERS): 
        p = torch.multiprocessing.Process(target=pusher_worker, 
                                args=(policy_net,replay_memory,
                                    current_episode, NUM_EPISODES,worker_num,))
        p.start()
        processes.append(p)
        time.sleep(0.01)


    ##Start sampler worker
    p = torch.multiprocessing.Process(target=sampler_worker, 
                           args=(policy_net, replay_memory, device,
                            current_episode, NUM_EPISODES,))
    p.start()
    processes.append(p)
    time.sleep(0.01)

    # join processes. The main purpose of join() is to ensure that a child process has completed before the main process does anything that depends on the work of the child process.
    for worker_num in range(len(processes)):
        p  = processes[worker_num]
        p.join()
        time.sleep(0.01)

    print('Done training')
    # Done training
