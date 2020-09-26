'''
TODO 
- save file with all recorded data so far, so that later I can 
play it back faster and see if I can improve training speed
- during eval print out not just the one that just evaled,
but all possible state action values on the tree, recursively called,
( can be used to get the probability of random selection of each?)



# Testing:/
    # read in terrain
    # for num samples:
        # for inner dqn loop:
            # select with boltzman sampling
            # record probability of that selection
        # output reward estimate for that design
    # take 5 most common robots and their reward estimates
         
'''


import torch
import torch.nn.functional as F

from design_assembler import module_types, num_module_types
from design_assembler import add_module, module_vector_list_to_robot_name
from simulation_runner import simulation_runner
from dqn import dqn
import random
import os
import logging
import numpy as np


cwd = os.path.dirname(os.path.realpath(__file__))

device = torch.device('cpu')
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

### hyperparameters

N_ACTIONS = num_module_types
MAX_N_MODULES = 3
NUM_ENVS = 3 # number of environments to run at the same time for a single terrain


### Initialize and load policy
sim_runner = simulation_runner(NUM_ENVS)
# sim_runner = simulation_runner(NUM_ENVS, show_GUI=True)
terrain_grid_shape = sim_runner.terrain_grid_shape
### Initialize DQN and replay buffer


folder = '09242020/'
folder = '09252020/'
# initiate the policy network and target network. 

PATH = os.path.join(cwd, folder+'policy_net_params.pt')
save_dict = dict()

save_dict = torch.load(PATH, map_location=lambda storage, loc: storage)

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

policy_net.eval()




# select an action randomly with boltzmann exploration
def select_multinomial_action(designs, terrains):
    with torch.no_grad():
        actions_out = policy_net(designs, terrains)
        # Scale by a temperature factor. as temperature gets lower, tends towards uniform outputs. as higher, tends towards true max. 
        actions_softmax = F.softmax(actions_out, dim=-1)

    action_inds = torch.multinomial(actions_softmax, 1, replacement=True)
    action_inds = action_inds.squeeze()
   
    return action_inds, actions_out, actions_softmax

# selects an action with the max Q value from all actions.
def select_max_action(designs, terrains):
    with torch.no_grad():
        actions_out = policy_net(designs, terrains)
        action_inds = actions_out.max(1)[1]
        return action_inds, actions_out 

def run_episode(terrains, use_max = True):

    n_samples = terrains.shape[0]

    # initialize empty robot batch
    designs = torch.zeros(n_samples, N_ACTIONS*MAX_N_MODULES + MAX_N_MODULES)
    designs[:, N_ACTIONS*MAX_N_MODULES] = 1 # indicate which port adding now
    
    # loop dqn until done:
    prob = torch.ones(n_samples)
    for i_dqn in range(MAX_N_MODULES):
        if use_max:
            actions, state_action_values = select_max_action(
                             designs.to(device),
                             terrains.to(device))
            prob = prob*1
        else:
            actions, state_action_values, actions_softmax = select_multinomial_action(
                             designs.to(device),
                             terrains.to(device))

            # multiply on the selection prob to get total prob so far for this design
            prob_mult = torch.gather(actions_softmax, 1, actions.unsqueeze(-1)).squeeze()
            # print('shape:') # torch.gather(input, dim, index,
            # print(actions.shape)
            # print(actions_softmax.shape)
            # print(prob_mult.shape)
            # print(prob.shape)

            prob = prob*prob_mult


        # add a module
        next_designs = torch.zeros_like(designs)
        for i_env in range(n_samples):
            next_designs[i_env,:] = add_module(
                                    designs[i_env,:], 
                                    i_dqn, MAX_N_MODULES,
                                    actions[i_env])


        if i_dqn==(MAX_N_MODULES-1): # we are done
            non_final = torch.tensor(0, dtype=torch.bool)

            # convert one-hot into list module names
            robot_names_list = []
            for i_env in range(n_samples):
                # print(next_designs[i_env])
                mv = next_designs[i_env,:N_ACTIONS*MAX_N_MODULES].reshape(MAX_N_MODULES,N_ACTIONS)
                # print(mv)
                robot_name = module_vector_list_to_robot_name(mv)
                robot_names_list.append(robot_name)

        else:
            non_final = torch.tensor(1, dtype=torch.bool)
        


        # print('designs')
        # print(str(designs.cpu().numpy()))
        # print('state_action_values')
        # print(str(state_action_values.cpu().numpy()))
        # print('Actions ' + str(actions.cpu().numpy()))
        # print('next_designs')
        # print(str(next_designs.cpu().numpy()))
        # print('-------------')

        # hold designs for next step
        designs = next_designs


    # print('names list:     ' + str(robot_names_list))
    # print('selection probs:'+ str(prob.numpy()))

    return robot_names_list, prob

terrain_block_heights = np.linspace(
        sim_runner.MAX_BLOCK_HEIGHT_LOW+0.01,
        sim_runner.MAX_BLOCK_HEIGHT_HIGH, 3)
for it in range(len(terrain_block_heights)):
    terrain_block_height = terrain_block_heights[it] 
    terrain = sim_runner.randomize_terrains(
        terrain_block_height=terrain_block_height)

    terrain_max = torch.max(terrain).numpy().item()
    # print('terrain means: ' + str(terrain_means))
    print('terrain max: ' + str(terrain_max))


    n_samples = 100
    terrains_rep = terrain.expand(n_samples,-1, -1, -1)
    # print(terrains.shape)
    # print(terrains_rep.shape)


    robot_names_out, prob = run_episode(terrains_rep,
             use_max=False) # for validation, don't simulate or store anything,
    # run with a range of terrains to check output

    # compare with a range of real robots:
    test_robot_list = ['lll', 'llw','lwl','wll',
                        'www', 'wwl', 'wlw', 'lww',
                        'lnl', 'lnw', 'wnl', 'wnw']
    # test_robot_list = ['lll', 'lwl', 'wnw']
    test_robot_rewards = []
    name_count = []
    name_indices = []
    name_probs = []
    for test_robot_name in test_robot_list:

        # terrains = sim_runner.randomize_terrains(
        #     terrain_block_height=terrain_block_height)

        sim_runner.load_robots(test_robot_name)
        r = sim_runner.run_sims(video_name_addition = str(it))
        # r = torch.tensor([1])
        test_robot_rewards.append(r.mean()) # mean of multiple runs

        indices = [i for i, x in enumerate(robot_names_out) if x == test_robot_name]
        # name_count.append(robot_names_out.count(test_robot_name))
        name_indices.append(indices)
        if len(indices)>0:
            name_probs.append(prob[indices[0]].numpy())
        else:
            name_probs.append(0)
        name_count.append(len(indices))

    name_count = np.array(name_count)
    name_probs = np.array(name_probs)
    # print(name_indices)
    test_robot_rewards = torch.cat(test_robot_rewards).numpy()
    print('Test robots:' + str(test_robot_list))
    print('Test rewards:' + str(test_robot_rewards))
    print('Test reward dist: ' + str(test_robot_rewards/test_robot_rewards.sum()))
    print('Output count dist:' + str(name_count/name_count.sum()))
    print('Output probs :    ' + str(name_probs))
    print('-----------')









