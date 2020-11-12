
import torch
from design_assembler import module_types, num_module_types
from design_assembler import add_module, module_vector_list_to_robot_name
from simulation_runner import simulation_runner, terrain_grid_shape
# from dqn_prev import dqn # older than 0929 use this
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

folder = 'mbrl_v6_test11/'
fname = 'policy_net_params.pt'
fname = 'policy_net_params (copy).pt'

folder = 'mamba_11112020/'
folder = 'mamba_11112020_2/'
fname = 'policy_net_params.pt'


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
    hidden_layer_size=save_dict['hidden_layer_size'],
    n_conv_layers = save_dict['n_conv_layers'])
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
    print('INITIALIZING (takes a few seconds)')
    sim_runner = simulation_runner(1, show_GUI= True, gui_speed_factor=2)

    start_sim_button = sim_runner.envs[0].p.addUserDebugParameter(
                        paramName = 'Start sim',
                        rangeMin = 0, rangeMax= -1,startValue =0)

    randomize_terrain_button = sim_runner.envs[0].p.addUserDebugParameter(
                        paramName = 'Randomize Terrain',
                        rangeMin = 0, rangeMax= -1,startValue =0)
    terrain_button_val = 0

    # sim_runner.record_video=True
    height_slider_ID = sim_runner.envs[0].p.addUserDebugParameter(
                            paramName = 'Terrain max height',
                            rangeMin = 0, rangeMax= 0.08,startValue =0)

    starting_density = 0.6
    density_slider_ID = sim_runner.envs[0].p.addUserDebugParameter(
                            paramName = 'Terrain sparsity',
                            rangeMin = 0.4, rangeMax= 0.9,startValue =starting_density)


    terrain = sim_runner.randomize_terrains(terrain_block_height=0.005,
        terrain_block_distance=starting_density)
    xyyaw = [0,0,0]


    sim_runner.envs[0].p.configureDebugVisualizer(sim_runner.envs[0].p.COV_ENABLE_GUI,1)
    # sim_runner.envs[0].follow_with_camera = True # follow the robot with camera

    robot_now = None
    robot_names_list, state_action_value =  run_episode(terrain)
    robot_now = robot_names_list[0] 

    sim_runner.load_robots(robot_now, 
        randomize_xyyaw=False, start_xyyaw = xyyaw)
    xyyaw = [sim_runner.envs[0].pos_xyz[0],
                sim_runner.envs[0].pos_xyz[1],
                sim_runner.envs[0].pos_rpy[2]]

    # wait for button started press to start sim
    started = False
    while not(started):
        button_read = sim_runner.envs[0].p.readUserDebugParameter(start_sim_button)
        if button_read >0:
            started = True

    # get initial debug parameters
    debug_params = [height_slider_ID]
    initial_debug_param_values = []
    if debug_params is not None:
        for param in debug_params:
            initial_debug_param_values.append(
                sim_runner.envs[0].p.readUserDebugParameter(param))
    last_terrain_value = 0

    while True:
        terrain_changed, terrain_value = sim_runner.step_envs([True], 
            [height_slider_ID],
            initial_debug_param_values)



        xyyaw = [sim_runner.envs[0].pos_xyz[0],
                    sim_runner.envs[0].pos_xyz[1],
                    sim_runner.envs[0].pos_rpy[2]]

        terrain_button_read = sim_runner.envs[0].p.readUserDebugParameter(randomize_terrain_button)
        if not(terrain_button_val==terrain_button_read):
            print('Reset terrain button press number ' + str(terrain_button_read))
            terrain_button_val = terrain_button_read
            density_slider_read = sim_runner.envs[0].p.readUserDebugParameter(density_slider_ID)
            print('Terrain density slider is at ' + str(density_slider_read))
            terrain = sim_runner.randomize_terrains(terrain_block_height=0.005,
                terrain_block_distance=density_slider_read)

            param_out = sim_runner.envs[0].p.readUserDebugParameter(height_slider_ID)

            sim_runner.terrain_randomizer.set_block_heights([sim_runner.envs[0].p],param_out)
            terrain_new = terrain.clone()
            terrain_new[terrain_new>0.001 ] += last_terrain_value
            robot_names_list, state_action_value =  run_episode(terrain_new)
            robot_now = robot_names_list[0]
            xyyaw = [0,0,0]
            sim_runner.load_robots(robot_now, 
                randomize_xyyaw=False, start_xyyaw = xyyaw)
            # param_out = sim_runner.step_simulation(debug_params= [height_slider_ID])
            terrain_changed = False

        # check if debug params have changed
        if terrain_changed and terrain_value is not None:
            terrain_new = terrain.clone()
            terrain_new[terrain_new>0.001 ] += terrain_value
            last_terrain_value = terrain_value

            robot_names_list, state_action_value =  run_episode(terrain_new)
            robot_new = robot_names_list[0]

            if not(robot_new == robot_now):
                robot_now = robot_new
                sim_runner.load_robots(robot_now, 
                    randomize_xyyaw=False, start_xyyaw = xyyaw)

        if xyyaw[0]>9:
            env_state = sim_runner.envs[0].get_state()
            env_state[0][0:2] = 0 
            sim_runner.envs[0].set_state(env_state)


    # import vid_cat_ffmpeg