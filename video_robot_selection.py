
import torch
from design_assembler import module_types, num_module_types
from design_assembler import add_module, module_vector_list_to_robot_name
from simulation_runner import simulation_runner, terrain_grid_shape
from dqn import dqn
import torch.nn.functional as F
import os
import numpy as np
cwd = os.path.dirname(os.path.realpath(__file__))
### hyperparameters


N_ACTIONS = num_module_types
MAX_N_MODULES = 3
device = torch.device('cpu')

### Initialize and load policy

folder = '09262020/'
PATH = os.path.join(cwd, folder+'policy_net_params.pt')
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


# selects an action with the max Q value from all actions.
def select_max_action(designs, terrains):
    with torch.no_grad():
        actions_out = policy_net(designs, terrains)
        action_inds = actions_out.max(1)[1]
        return action_inds, actions_out 

def run_episode(terrains):

    n_samples = terrains.shape[0]

    # initialize empty robot batch
    designs = torch.zeros(n_samples, N_ACTIONS*MAX_N_MODULES + MAX_N_MODULES)
    designs[:, N_ACTIONS*MAX_N_MODULES] = 1 # indicate which port adding now
    
    # loop dqn until done:
    for i_dqn in range(MAX_N_MODULES):
        actions, state_action_values = select_max_action(
                         designs.to(device),
                         terrains.to(device))

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

    return robot_names_list, state_action_value


if __name__== "__main__":
    sim_runner = simulation_runner(1, show_GUI= True)
    terrain = sim_runner.randomize_terrains(terrain_block_height=0.01,
        terrain_block_distance=0.7)
    xyyaw = [0,0,0]
    sim_runner.record_video=True
    # slider_ID = sim_runner.envs[0].p.addUserDebugParameter(
    #                         paramName = 'Terrain max height',
    #                         rangeMin = 0, rangeMax= 0.1,startValue =0)

    num_increases = 3
    robot_now = None
    for i in range(num_increases):
        terrain = sim_runner.measure_terrains()
        robot_names_list, state_action_value =  run_episode(terrain)
        robot_now = robot_names_list[0] 
        sim_runner.load_robots(robot_now, 
            randomize_xyyaw=False, start_xyyaw = xyyaw)
        r = sim_runner.run_sims(n_time_steps=50, 
            video_name_addition='_livetest_' + str(i),
            debug_params= [slider_ID])
        xyyaw = [sim_runner.envs[0].pos_xyz[0],
            sim_runner.envs[0].pos_xyz[1],
            sim_runner.envs[0].pos_rpy[2]]
        print(r)
        if i<(num_increases-1):
            sim_runner.alter_terrain_height(0.015)

    # import vid_cat_ffmpeg