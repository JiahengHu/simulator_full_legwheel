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
import logging



cwd = os.path.dirname(os.path.realpath(__file__))

device = torch.device('cpu')
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

### hyperparameters

N_ACTIONS = num_module_types
MAX_N_MODULES = 3
NUM_ENVS = 3 # number of environments to run at the same time for a single terrain


def run_test(folder, fname):

    ### Initialize and load policy
    sim_runner = simulation_runner(NUM_ENVS)
    # sim_runner = simulation_runner(NUM_ENVS, show_GUI=True)
    terrain_grid_shape = sim_runner.terrain_grid_shape
    ### Initialize DQN and replay buffer


    # folder = '09262020/'
    # PATH = os.path.join(cwd, folder+'policy_net_params.pt')


    # PATH = os.path.join(cwd, folder+'policy_net_params.pt')
    PATH = os.path.join(cwd, folder+fname)


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


    log_path =  os.path.join(cwd, folder+"results.log")
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s', 
                        filename=log_path,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    print = logging.info

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
                next_designs[i_env,:], penalty = add_module(
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
                
                state_action_value = torch.gather(state_action_values, 1, actions.unsqueeze(-1)).squeeze()

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

        return robot_names_list, prob, state_action_value

    terrain_block_heights = np.linspace(
            sim_runner.MAX_BLOCK_HEIGHT_LOW+0.01,
            sim_runner.MAX_BLOCK_HEIGHT_HIGH, 3)
    for it in range(len(terrain_block_heights)):
        terrain_block_height = terrain_block_heights[it] 
        terrain = sim_runner.randomize_terrains(
            terrain_block_height=terrain_block_height,
            terrain_block_distance=0.6)
        # print('Terrain seed: ' + str(sim_runner.terrain_seed))
        # print('Terrain grid: ')
        # print(sim_runner.terrain_grid)

        terrain_max = torch.max(terrain).numpy().item()
        # print('terrain means: ' + str(terrain_means))
        print('terrain max: ' + str(terrain_max))


        n_samples = 1000
        terrains_rep = terrain.expand(n_samples,-1, -1, -1)
        # print(terrains.shape)
        # print(terrains_rep.shape)

        print('running DQN')
        robot_names_out, prob, state_action_value = run_episode(terrains_rep,
                 use_max=False) # for validation, don't simulate or store anything,
        # run with a range of terrains to check output

        # compare with a range of real robots:
        test_robot_list = ['lll', 'llw','lwl','wll',
                            'www', 'wwl', 'wlw', 'lww',
                            'lnl', 'lnw', 'wnl', 'wnw']
        # test_robot_list = ['lll', 'lwl', 'wnw']
        name_count = []
        name_indices = []
        name_probs = []
        name_values = []

        test_robot_str = ''
        for rn in test_robot_list:
            test_robot_str += rn
            test_robot_str += ' '
        print(' Robots: ' + str(test_robot_str))

        for test_robot_name in test_robot_list:

            indices = [i for i, x in enumerate(robot_names_out) 
                        if x == test_robot_name]
            # name_count.append(robot_names_out.count(test_robot_name))
            name_indices.append(indices)
            if len(indices)>0:
                name_probs.append(prob[indices[0]].numpy())
                name_values.append(state_action_value[indices[0]].numpy().item())
            else:
                name_probs.append(0)
                name_values.append(0)
            name_count.append(len(indices))

        name_count = np.array(name_count)
        name_probs = np.array(name_probs)
        # print(name_indices)
        print('Output count dist:' + np.array2string(
                                name_count/np.sum(name_count),
                                precision=2))
        print('Output probs :    ' + np.array2string(
                                name_probs,precision=2))
        name_values_str = '['
        for v in name_values:
            if v is not None:
                name_values_str += np.array2string(np.array(v),
                    precision=2)
            else:
                name_values_str += 'N/A'
            name_values_str += ' '
        name_values_str += ']'
        print('Output values :    ' + str(name_values_str))
        test_robot_rewards = []
        test_robot_rewards_min = []
        test_robot_rewards_max = []

        print('running Simulations')
        for test_robot_name in test_robot_list:

            # terrains = sim_runner.randomize_terrains(
            #     terrain_block_height=terrain_block_height)

            sim_runner.load_robots(test_robot_name)
            r = sim_runner.run_sims(video_name_addition = str(it))
            # r = torch.tensor([1])
            test_robot_rewards.append(r.mean().numpy().item()) # mean of multiple runs
            test_robot_rewards_min.append(r.min().numpy().item()) # mean of multiple runs
            test_robot_rewards_max.append(r.max().numpy().item()) # mean of multiple runs


        test_robot_rewards = np.array(test_robot_rewards)
        test_robot_rewards_min = np.array(test_robot_rewards_min)
        test_robot_rewards_max = np.array(test_robot_rewards_max)
        print('Sim rewards:' + np.array2string(
                                test_robot_rewards,precision=2))
        print('Sim rewards min:' + np.array2string(
                                test_robot_rewards_min,precision=2))
        print('Sim rewards max:' + np.array2string(
                                test_robot_rewards_max,precision=2))
        print('Sim reward dist: ' + np.array2string(
                                test_robot_rewards/np.sum(test_robot_rewards),
                                precision=2))

        print('-----------')
        print('Estimated top 5:')
        val_inds = np.argsort(name_values)[::-1]
        name_str = ''
        est_name_list = []
        val_str = ''
        for i in range(5):
            name_str += test_robot_list[val_inds[i]] + ','
            est_name_list.append(test_robot_list[val_inds[i]])
            val_str += np.array2string(np.array(name_values[val_inds[i]]),
                                        precision=1) + ','
        print('Names: '  +name_str)
        print('Values: ' +val_str)

        print('Simulated top 5:')
        val_inds = np.argsort(test_robot_rewards)[::-1]
        name_str = ''
        sim_name_list = []
        val_str = ''

        for i in range(5):
            name_str += test_robot_list[val_inds[i]] + ','
            sim_name_list.append(test_robot_list[val_inds[i]])
            val_str += np.array2string(np.array(test_robot_rewards[val_inds[i]]),precision=1) 
            val_str += ' [' + np.array2string(np.array(test_robot_rewards_min[val_inds[i]]),precision=1)
            val_str +=  ' - ' + np.array2string(np.array(test_robot_rewards_max[val_inds[i]]),precision=1)
            val_str +=  '],'
        print('Names: '  +name_str)
        print('Values: ' +val_str)

        overlap = 0
        for sn in sim_name_list:
            if sn in est_name_list:
                overlap+=1
        print('Overlap: ' + str(overlap) )
        print('-----------')



if __name__== "__main__":

    folder = '09282020_mamba/'
    folder = '09282020_viper/'
    folder = '10012020_viper/'
    fname = 'policy_net_params_par.pt'
    run_test(folder,fname)



