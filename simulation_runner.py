'''
loads the policy gnn and contains multiple 
simulation environments

'''

# load libraries
import torch
from robot_env import robot_env
import numpy as np
import pgnn_control as pgnnc
from utils import to_tensors, combine_state, wrap_to_pi, rotate, create_control_inputs
from collections import deque
from poisson_terrain import TerrainRandomizer
import os
cwd = os.path.dirname(os.path.realpath(__file__))

device = torch.device("cpu")

np.set_printoptions(precision=2,suppress=True)

class simulation_runner(object):

    def __init__(self, num_envs = 1, show_GUI=False):
        self.num_envs = num_envs
        # self.terrain_grid_shape = [1,20,10] # deltax, deltay
        self.terrain_grid_shape = [1,50,20] # deltax, deltay
        self.robot_names_list = []
        self.terrain_grid = []
        self.max_xy = [9,2] # defines size of obstacle course. TODO: get from terrain randomizer.
        self.reward_function = 'Simulation'
        # self.reward_function = 'Recorded Simulation'
        # self.reward_function = 'Testing Proxy'
        self.modules_list = []
        self.attachments_list = []
        self.module_action_len_list = []
        self.saved_data = None
        self.normal_distribution = torch.distributions.normal.Normal(0, 1)

        # terrain parameters:
        self.terrain_randomizer = TerrainRandomizer()
        self.show_GUI = show_GUI # if true, will show first gui. for test only.
        # self.show_GUI = True # if true, will show first gui. for test only.
        self.record_video= False # if show gui and true, records some videos
        self.record_video= True # if show gui and true, records some videos
        # obstacle density and size
        # high and large and dense
        self.MIN_BLOCK_DISTANCE_LOW = 0.4
        self.MAX_BLOCK_HEIGHT_HIGH = 0.1
        # shallow and small and sparse
        self.MIN_BLOCK_DISTANCE_HIGH = 1.
        self.MAX_BLOCK_HEIGHT_LOW = 0
        self.terrain_scaling = 1. # 0: no bumps. 1: max bumps

        # points for grid height with ray casting
        # self.terrain_grid_shape = [1,20,10]
        grid_x, grid_y = np.meshgrid(
            np.linspace(-1,self.max_xy[0],self.terrain_grid_shape[1]),
             np.linspace(-self.max_xy[1],self.max_xy[1],self.terrain_grid_shape[2]))
        self.rayFromPositions = []
        self.rayToPositions = []
        for x,y in zip(grid_x.flatten(), grid_y.flatten()):
            self.rayFromPositions.append([x,y,10])
            self.rayToPositions.append([x,y,-1])


        # create GNN
        # load GNN params
        if self.reward_function == 'Simulation':

            PATH = 'mbrl_v4_test17/multidesign_control_iter3.pt'
            PATH = 'mbrl_v4_test18/multidesign_control_iter3_v4.pt'
            PATH = os.path.join(cwd, PATH)
            save_dict = torch.load( PATH, map_location=lambda storage, loc: storage)
            goal_len =3
            self.gnn_nodes = pgnnc.create_GNN_nodes(save_dict['internal_state_len'] , 
                                       save_dict['message_len'] , 
                                       save_dict['hidden_layer_size'],
                                       device, goal_len=goal_len, body_input = True)
            pgnnc.load_state_dicts(self.gnn_nodes, save_dict['gnn_state_dict']) 


            # create environments
            self.envs = []
            for i_env in range(num_envs):
                # init env
                # use the env to get the size of the inputs and outputs
                if i_env==0:
                    env = robot_env(show_GUI = self.show_GUI)
                else:
                    env = robot_env(show_GUI = False)
                # env.reset_terrain() # todo: remove when randomize is done
                self.envs.append(env)

                # env.p.resetDebugVisualizerCamera(3,0,-89.999,[3,0,0.2],physicsClientId=env.physicsClient) 
                env.p.resetDebugVisualizerCamera(2.5,0,-30,[2.5,0,0.2],physicsClientId=env.physicsClient) 

                env.sim_speed_factor = 2

        elif self.reward_function == 'Recorded Simulation':
            # load up rewards to look up
            self.saved_data = torch.load('rewards_processed.pt')




    def measure_terrains(self):
        terrain_grid = torch.zeros([self.num_envs]+self.terrain_grid_shape)

        if self.reward_function == 'Testing Proxy' or self.reward_function == 'Recorded Simulation':
            # for testing:
            for i_env in range(self.num_envs):
                if np.random.rand()>0.5:
                    terrain_grid[i_env] += 0.1 # make terrain entries large sometimes for test
            self.terrain_grid = terrain_grid

        elif self.reward_function == 'Simulation':
            # using ray casting to get heights of points

            for i_env in range(self.num_envs):

                rays_batch = self.envs[i_env].p.rayTestBatch(
                    rayFromPositions = self.rayFromPositions,
                    rayToPositions = self.rayToPositions)
                n_pts = self.terrain_grid_shape[1]*self.terrain_grid_shape[2]
                heights = torch.zeros(n_pts)
                for i_ray in range(n_pts):
                    # extract z world height of raycast
                    heights[i_ray] = rays_batch[i_ray][3][-1]

                terrain_grid[i_env] =  heights.reshape(self.terrain_grid_shape)
                # print('terrain ' + str(i_env) + ': ---')
                # print(terrain_grid[i_env].numpy())

        self.terrain_grid = terrain_grid

        return self.terrain_grid

    def randomize_terrains(self,
        terrain_block_height = None,
        terrain_block_distance=None):
        # randomize the terrains


        if self.reward_function == 'Simulation':


            for i_env in range(self.num_envs):

                # make videos where all robots have the same terrain
                if self.show_GUI and self.record_video:
                    np.random.seed(0)
                    torch.manual_seed(0)


                self.envs[i_env].reset_terrain() # this seems to fix the memory leak

                # randomizes terrain with poisson sampling blocks
                self.terrain_randomizer.reset()

                if terrain_block_height is None:
                    terrain_block_height   = np.random.uniform(self.MAX_BLOCK_HEIGHT_LOW,
                             self.MAX_BLOCK_HEIGHT_LOW + (self.MAX_BLOCK_HEIGHT_HIGH - self.MAX_BLOCK_HEIGHT_LOW) * self.terrain_scaling)
                if terrain_block_distance is None:
                    terrain_block_distance = np.random.uniform(
                         self.MIN_BLOCK_DISTANCE_HIGH + (self.MIN_BLOCK_DISTANCE_LOW - self.MIN_BLOCK_DISTANCE_HIGH) * self.terrain_scaling,
                         self.MIN_BLOCK_DISTANCE_HIGH
                         )
                self.terrain_block_distance = terrain_block_distance
                self.terrain_block_height = terrain_block_height
                self.terrain_randomizer.randomize_env(self.envs[i_env].p,
                    terrain_block_distance, terrain_block_height)


        # measure their heights
        terrains = self.measure_terrains()

        return terrains

    def check_robot_validity(self, urdf_name):
        is_valid = True
        # at most two n
        if urdf_name.count('n')>2:
            is_valid = False
        # do not allow n in front or back
        if (urdf_name[0]=='n' or 
            urdf_name[2]=='n' or
            urdf_name[3]=='n' or  
            urdf_name[5]=='n'):
            is_valid = False
        return is_valid


    def load_robots(self,robot_names_list):
        self.robot_names_list = robot_names_list
        # load in each robot to the environments


        # these are the parts we need to save for later
        self.modules_list = []
        self.is_valid = []

        for i_env in range(self.num_envs):
            robot_name = robot_names_list[i_env]

            urdf_name = robot_name + robot_name[::-1] # for symmetric designs

            is_valid = self.check_robot_validity(urdf_name)
            self.is_valid.append(is_valid)

            if is_valid and self.reward_function == 'Simulation':

                # env = self.envs[i_env]
                # env.reset_robot(urdf_name=urdf_name, randomize_start=False)
                # ### THIS IS CAUSING A MEMORY LEAK_-- WHY???
                # attachments = env.attachments
                # modules_types = env.modules_types
                # n_modules = len(modules_types)

                env = self.envs[i_env]
                env.reset_robot(urdf_name=urdf_name, randomize_start=False)
                attachments = env.attachments
                modules_types = env.modules_types
                n_modules = len(modules_types)


                # create module containers for the nodes
                modules = []
                for i in range(n_modules):
                    modules.append(pgnnc.Module(i, self.gnn_nodes[modules_types[i]],
                                     device))
                module_action_len = list(np.diff(env.action_indexes))

                self.modules_list.append(modules)

            else:
                self.modules_list.append(None)



    def run_sims(self,n_time_steps=150, video_name_addition = ''):
    # runs simulations and returns how far they went
    #     for 100 time steps
    #         get robot position
    #         compute new heading to steer back toward center and forward 
    #     get final displacement as reward out
    # todo: set up a batch pool to run simulations with runner


        rewards = torch.zeros(self.num_envs, dtype=torch.float32)


        if self.reward_function == 'Simulation':
            for i_env in range(self.num_envs):


                if not(self.is_valid[i_env]):
                    rewards[i_env] -= 10
                else:
                    rewards[i_env] += simulate_robot( 
                            self.envs[i_env], 
                            self.modules_list[i_env],
                            self.record_video,
                            self.max_xy,
                            video_name_addition)


        elif self.reward_function == 'Recorded Simulation':
            for i_env in range(self.num_envs):
                if not(self.is_valid[i_env]):
                    rewards[i_env] -= 10
                else:
                    key = self.robot_names_list[i_env]
                    rewards[i_env] += self.saved_data[key]
                    # add artificial noise
                    rewards[i_env] += self.normal_distribution.sample()

        elif self.reward_function == 'Testing Proxy':
            # # # ----------- placeholder reward ---------------
            for i_env in range(self.num_envs):
            # alter rewards based on terrain
                if torch.max(self.terrain_grid[i_env])>0.05:
                    w_reward = 1
                    l_reward = 2
                else:
                    l_reward = 1
                    w_reward = 2

                robot_name = self.robot_names_list[i_env]
                for letter in robot_name:
                    if letter=='l':
                        rewards[i_env]+=l_reward
                    elif letter=='w':
                        rewards[i_env]+=w_reward


        return rewards



def simulate_robot( env, modules, record_video, max_xy, video_name_addition):
    n_time_steps=150
    module_action_len= list(np.diff(env.action_indexes))
    attachments = env.attachments
    n_modules = len(env.modules_types)
    pos_queue = deque(maxlen=20)

    reward = 0

    # subtract out initial x, in case its not exactly zero
    reward -= env.pos_xyz[0]
    logID = None
    if env.show_GUI and record_video:
        vid_path = os.path.join(cwd, 
                    env.loaded_urdf+ video_name_addition+'.mp4')

        if not os.path.exists(vid_path):
            logID = env.p.startStateLogging(
                env.p.STATE_LOGGING_VIDEO_MP4,
                fileName=vid_path)
            

    for step in range(n_time_steps):
        chassis_yaw = env.pos_rpy[-1]
        chassis_x = env.pos_xyz[0]
        chassis_y = env.pos_xyz[1]
        
        # set direction to head
        desired_xyyaw = np.zeros(3)
        desired_xyyaw[0] = 1.5
        desired_xyyaw[1] = -1.5*chassis_y
        desired_xyyaw[1] = np.clip(desired_xyyaw[1], -1.5,1.5)
        # desired_xyyaw[2] = -2.5*chassis_yaw
        # desired_xyyaw[2] = np.clip(desired_xyyaw[2], -1.5,1.5)


        env_state = env.get_state()
        states = [smm.to(device) for smm in to_tensors(env_state)]

        goals_world = torch.tensor(desired_xyyaw, 
                dtype=torch.float32, device=device).unsqueeze(0)
        node_inputs = create_control_inputs(states, goals_world)

        for module in modules: # this prevents the LSTM in the GNN nodes from 
            # learning relations over time, only over internal prop steps.
            module.reset_hidden_states(1) 

        with torch.no_grad():
            out_mean, out_var = pgnnc.run_propagations(
                modules, attachments, 2, node_inputs, device)
            u_out_mean = []
            tau_out_mean = []
            for mm in range(n_modules):
                u_out_mean.append(out_mean[mm][:,:module_action_len[mm]])
                tau_out_mean.append(out_mean[mm][:,module_action_len[mm]:])
            u_np = torch.cat(u_out_mean,-1).squeeze().numpy()
        env.step(u_np)
        
        # check if flipped
        if np.dot([0,0,1], env.z_axis)<0:
            break

        # check if out of obstacle course
        # if (np.abs(env.pos_xyz[0])>max_xy[0] or 
        #     np.abs(env.pos_xyz[1])>max_xy[1]):
        #     break

        if np.abs(env.pos_xyz[1])>max_xy[1]:
            break


        # check if stuck
        pos_queue.append(np.array([env.pos_xyz[0], env.pos_xyz[1]]))
        if len(pos_queue) == pos_queue.maxlen:
            delta_pos = np.linalg.norm(pos_queue[-1] - pos_queue[0])
            if delta_pos<0.02: # threshold for being considered stuck
                break

    reward += env.pos_xyz[0]


    if env.show_GUI and (logID is not None):
        env.p.stopStateLogging(logID)

    return reward

