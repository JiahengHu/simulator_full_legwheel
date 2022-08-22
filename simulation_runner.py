'''
loads the policy gnn and contains multiple 
simulation environments

todo: run a few of the same robots in the same env each time, using
batches in the policy, to get an expectation of reward for the robot.


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
from datetime import datetime

time_program = False

cwd = os.path.dirname(os.path.realpath(__file__))

device = torch.device("cpu")

np.set_printoptions(precision=2,suppress=True)

terrain_grid_shape = [1,50,20]
# MAX_BLOCK_HEIGHT_HIGH = 0.1
MAX_BLOCK_HEIGHT_HIGH = 0.08
MIN_BLOCK_DISTANCE_LOW = 0.4
MIN_BLOCK_DISTANCE_HIGH =1.
MAX_BLOCK_HEIGHT_LOW = 0
reward_function = 'Simulation'
# reward_function = 'Recorded Simulation'
# reward_function = 'Testing Proxy'
# control_file = 'mbrl_v6_test10/multidesign_control_iter4.pt'
# control_file = 'mbrl_v6_test11/multidesign_control_iter4.pt'
control_file = 'mbrl_b_v8_test5/multidesign_control_iter3.pt'
control_file = 'multidesign_control_iter3_b_tripod/multidesign_control_iter3_b_tripod.pt'
neural_policy = True

class simulation_runner(object):

    def __init__(self, num_envs = 1, show_GUI=False, record_video=False,
        gui_speed_factor = 1):
        self.num_envs = num_envs
        # self.terrain_grid_shape = [1,20,10] # deltax, deltay
        # self.terrain_grid_shape = [1,50,20] # deltax, deltay
        self.terrain_grid_shape = terrain_grid_shape # deltax, deltay
        self.robot_name = [] # all envs have the same robot loaded
        self.terrain_grid = []
        self.max_xy = [9,2] # defines size of obstacle course. TODO: get from terrain randomizer.

        self.modules_gnn = []
        self.saved_data = None
        self.normal_distribution = torch.distributions.normal.Normal(0, 1)

        # terrain parameters:
        self.terrain_randomizer = TerrainRandomizer()
        self.show_GUI = show_GUI # if true, will show first gui. for test only.
        # self.show_GUI = True # if true, will show first gui. for test only.
        self.record_video = record_video # if show gui and true, records some videos
        # self.record_video= True # if show gui and true, records some videos
        # obstacle density and size
        # high and large and dense
        self.MIN_BLOCK_DISTANCE_LOW = MIN_BLOCK_DISTANCE_LOW
        self.MAX_BLOCK_HEIGHT_HIGH = MAX_BLOCK_HEIGHT_HIGH
        # self.MAX_BLOCK_HEIGHT_HIGH = 0.075
        # shallow and small and sparse
        self.MIN_BLOCK_DISTANCE_HIGH = MIN_BLOCK_DISTANCE_HIGH
        self.MAX_BLOCK_HEIGHT_LOW = MAX_BLOCK_HEIGHT_LOW
        self.terrain_scaling = 1. # 0: no bumps. 1: max bumps
        self.terrain_seed = None
        self.control_file_loaded = None
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
        if reward_function == 'Simulation':

            # PATH = 'mbrl_v6_test10/multidesign_control_iter4.pt'
            PATH = control_file
            PATH = os.path.join(cwd, PATH)
            self.control_file_loaded = PATH
            save_dict = torch.load( PATH, map_location=lambda storage, loc: storage)
            print('Loaded network ' + PATH)
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
                env.follow_with_camera = True
                # env.p.resetDebugVisualizerCamera(3,0,-89.999,[3,0,0.2],physicsClientId=env.physicsClient) 
                # env.sim_speed_factor = 5
                env.sim_speed_factor = gui_speed_factor

                self.envs.append(env)

        elif reward_function == 'Recorded Simulation':
            # load up rewards to look up
            self.saved_data = torch.load('rewards_processed.pt')


    def save_img(self, index, path):
        p = self.envs[0].p
        width = 6
        height = 3
        x_offset = 3
        y_offset = 0
        # camera
        pixelWidth = 400
        pixelHeight = int(pixelWidth * height / width)
        nearPlane = 0.05
        farPlane = 15
        fov = 60
        aspect = pixelWidth / pixelHeight

        # calculate the camera distance
        camera_dist = height / 2 / np.tan(np.radians(fov / 2))
        # print(f"camera distance is {camera_dist}")

        # Initial vectors
        camera_vector = (0, 0, -1)  # z-axis
        up_vector = (0, 1, 0)  # y-axis
        camPos = (x_offset, y_offset, camera_dist)
        viewMatrix = p.computeViewMatrix(camPos, np.asarray(camPos) + 0.1 * np.asarray(camera_vector), up_vector)
        projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)

        img = p.getCameraImage(pixelWidth,
                                   pixelHeight,
                                   viewMatrix,
                                   projectionMatrix,
                                   shadow=1,
                                   lightDirection=[1, 1, 1])[2]
        from PIL import Image
        im = Image.fromarray(img).convert('RGB')
        im.save(path+str(index), "jpeg")


    def measure_terrains(self):
        terrain_grid = torch.zeros([1]+self.terrain_grid_shape)

        if reward_function == 'Testing Proxy' or reward_function == 'Recorded Simulation':
            # for testing:
            if np.random.rand()>0.5:
                terrain_grid += 0.1 # make terrain entries large sometimes for test
            self.terrain_grid = terrain_grid

        elif reward_function == 'Simulation':
            # using ray casting to get heights of points

            # NOTE: Assumes that the terrains in the envs are all the same.
            
            # i_env = 0

            # for i_env in range(self.num_envs):
            for i_env in range(1): # since the envs are all the same,
            # only need to measure them once

                rays_batch = self.envs[i_env].p.rayTestBatch(
                    rayFromPositions = self.rayFromPositions,
                    rayToPositions = self.rayToPositions)
                n_pts = self.terrain_grid_shape[1]*self.terrain_grid_shape[2]
                heights = torch.zeros(n_pts)
                for i_ray in range(n_pts):
                    # extract z world height of raycast
                    heights[i_ray] = rays_batch[i_ray][3][-1]
                terrain_height = heights.reshape(self.terrain_grid_shape)

                terrain_grid[0] =  terrain_height
                # print('terrain : ---')
                # print(terrain_height[0].max(dim=1)[0])

        self.terrain_grid = terrain_grid

        return self.terrain_grid

    def randomize_terrains(self,
                           terrain_block_height=None,
                           terrain_block_distance=None, fixed_terrain=False, rd_seed=None):
        # randomize the terrains

        # ########## tmp ##############
        if fixed_terrain:
            new_seed = 0
            np.random.seed(new_seed)
        elif rd_seed is not None:
            np.random.seed(rd_seed)

        if reward_function == 'Simulation':

            # pick a random seed for which all the terrains will use,
            # so that they all get the same new terrain
            if self.show_GUI and self.record_video:
                new_seed = 0
                np.random.seed(new_seed)
                torch.manual_seed(new_seed)
                # make videos where all robots have the same terrain
            # else:
            #     new_seed = np.random.randint(4294967295)

            if terrain_block_height is None:
                terrain_block_height = np.random.uniform(self.MAX_BLOCK_HEIGHT_LOW,
                                                         self.MAX_BLOCK_HEIGHT_LOW + (
                                                                     self.MAX_BLOCK_HEIGHT_HIGH - self.MAX_BLOCK_HEIGHT_LOW) * self.terrain_scaling)
            if terrain_block_distance is None:
                terrain_block_distance = np.random.uniform(
                    self.MIN_BLOCK_DISTANCE_HIGH + (
                                self.MIN_BLOCK_DISTANCE_LOW - self.MIN_BLOCK_DISTANCE_HIGH) * self.terrain_scaling,
                    self.MIN_BLOCK_DISTANCE_HIGH
                )

            for i_env in range(self.num_envs):
                self.envs[i_env].reset_terrain()  # this seems to fix the memory leak

            # randomizes terrain with poisson sampling blocks
            self.terrain_randomizer.reset()

            self.terrain_block_distance = terrain_block_distance
            self.terrain_block_height = terrain_block_height
            # all terrains will get the same block added
            self.terrain_randomizer.randomize_env(
                [self.envs[i_env].p for i_env in range(self.num_envs)],
                terrain_block_distance, terrain_block_height)
            # print(terrain_block_distance)
            # print(terrain_block_height)

        # measure their heights
        terrains = self.measure_terrains()

        # make sure in the end we reset random seed
        np.random.seed()

        return terrains

    # def alter_terrain_height(self, delta_h):
    #     for i_env in range(self.num_envs):
    #         self.envs[i_env].reset_terrain() # this seems to fix the memory leak

    #     self.terrain_randomizer.alter_block_heights(
    #         [self.envs[i_env].p for i_env in range(self.num_envs)],
    #         delta_h)

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


    def load_robots(self,robot_name, randomize_xyyaw=True, start_xyyaw= None):
        self.robot_name = robot_name
        # load in each robot to the environments


        # these are the parts we need to save for later
        self.modules_list = []
        self.is_valid = []
        if len(robot_name) == 3:
            urdf_name = robot_name + robot_name[::-1] # for symmetric designs
        else:
            urdf_name = robot_name
        is_valid = self.check_robot_validity(urdf_name)


        self.is_valid = is_valid

        if is_valid and reward_function == 'Simulation':

            for i_env in range(self.num_envs):
                # env = self.envs[i_env]
                # env.reset_robot(urdf_name=urdf_name, randomize_start=False)
                # ### THIS IS CAUSING A MEMORY LEAK_-- WHY???
                # Need to reset sim periodically... it does not destroy removed objects from memory
                # attachments = env.attachments
                # modules_types = env.modules_types
                # n_modules = len(modules_types)

                env = self.envs[i_env]
                env.reset_robot(urdf_name=urdf_name, 
                    randomize_xyyaw=randomize_xyyaw,  # adds a little random perturbation to initial state
                    start_xyyaw=start_xyyaw, randomize_start=False)
                # add a small amount of noise onto the robot start pose

            modules_types = env.modules_types
            n_modules = len(modules_types)

            # create module containers for the nodes
            modules = []
            for i in range(n_modules):
                modules.append(pgnnc.Module(i, self.gnn_nodes[modules_types[i]],
                                 device))

            self.modules = modules

        else:
            self.modules =None

               



    def run_sims(self,n_time_steps=250, video_name_addition = '', 
        debug_params=None, allow_early_stop=True):
    # runs simulations and returns how far they went
    #     for 100 time steps
    #         get robot position
    #         compute new heading to steer back toward center and forward 
    #     get final displacement as reward out
    # todo: set up a batch pool to run simulations with runner

        if time_program:
            sim_start = datetime.now()

        rewards = torch.zeros(self.num_envs, dtype=torch.float32)
        power = torch.zeros(self.num_envs, dtype=torch.float32)

        if reward_function == 'Simulation':


            if not(self.is_valid):
                rewards -= 10
                power += 10
            else:
                r, p = self.simulate_robot(
                        video_name_addition,
                        n_time_steps,
                        debug_params,
                        allow_early_stop)
                rewards += r
                power += p


        elif reward_function == 'Recorded Simulation':
            for i_env in range(self.num_envs):
                if not(self.is_valid[i_env]):
                    rewards[i_env] -= 10
                else:
                    key = self.robot_name
                    rewards[i_env] += self.saved_data[key]
                    # add artificial noise
                    rewards[i_env] += self.normal_distribution.sample()

        elif reward_function == 'Testing Proxy':
            # # # ----------- placeholder reward ---------------
            for i_env in range(self.num_envs):
            # alter rewards based on terrain
                if torch.max(self.terrain_grid)>0.05:
                    w_reward = 1
                    l_reward = 2
                else:
                    l_reward = 1
                    w_reward = 2

                robot_name = self.robot_name
                for letter in robot_name:
                    if letter=='l':
                        rewards[i_env]+=l_reward
                    elif letter=='w':
                        rewards[i_env]+=w_reward

        if time_program:
            print(f"whole run_sim takes {datetime.now() - sim_start}")

        return rewards, power



    def simulate_robot(self,
        video_name_addition='', n_time_steps=200, 
         debug_params = None,
         allow_early_stop = True):


        envs = self.envs 
        record_video = self.record_video
        max_xy = self.max_xy
        terrain_randomizer = self.terrain_randomizer

        # NOTE: assumes that all envs have the same robot loaded
        num_envs = len(envs)
        

        pos_queues = [deque(maxlen=25)]*num_envs
        reward = torch.zeros(num_envs, dtype=torch.float32)
        power = torch.zeros(num_envs, dtype=torch.float32)
        # get initial debug parameters
        initial_debug_param_values = []
        if debug_params is not None:
            for param in debug_params:
                initial_debug_param_values.append(envs[0].p.readUserDebugParameter(param))

        for env in envs:
            # subtract out initial x, in case its not exactly zero
            reward -= env.pos_xyz[0]
            logID = None
            if env.show_GUI and record_video:
                vid_path = os.path.join(cwd, 
                            env.loaded_urdf+ video_name_addition+'.mp4')

                # if not os.path.exists(vid_path):
                logID = env.p.startStateLogging(
                    env.p.STATE_LOGGING_VIDEO_MP4,
                    fileName=vid_path)
                
        robot_alive = [True]*num_envs
        for step in range(n_time_steps):

            self.step_envs(robot_alive, debug_params, initial_debug_param_values)
            if allow_early_stop:

                for i_env in range(num_envs):
                    if robot_alive[i_env]:
                        env = envs[i_env]
                        pos_queue = pos_queues[i_env]
                        # check if flipped
                        if np.dot([0,0,1], env.z_axis) < 0.1:
                            robot_alive[i_env] = False

                        # check if out of obstacle course
                        if np.abs(env.pos_xyz[1])>max_xy[1]:
                            robot_alive[i_env] = False

                        # check if stuck
                        pos_queue.append(
                            np.array([env.pos_xyz[0], env.pos_xyz[1]]))
                        if len(pos_queue) == pos_queue.maxlen:
                            delta_pos = np.linalg.norm(pos_queue[-1] - pos_queue[0])
                            if delta_pos<0.02: # threshold for being considered stuck
                                robot_alive[i_env] = False


        for i_env in range(num_envs):
            env = envs[i_env]
            reward[i_env] += env.pos_xyz[0]
            try:
                power[i_env] += env.power
            except:
                pass
                # print("Warning: power is not implemented")
            if env.show_GUI and (logID is not None):
                env.p.stopStateLogging(logID)
        return reward, power

    def cal_wheel_ctl(self, env):
        pi = np.pi
        env_state_i = env.get_state()
        cur = []
        pointer = 0
        for module in env.ctl_modules_types:
            if module == 2:
                cur.append(env_state_i[1:][pointer][0])
            else:
                cur.append(0)
            if module != 0: #if null, don't add pointer
                pointer += 1

        target = np.array([-pi / 3, 0, pi / 3, -pi / 3, 0, pi / 3])
        diff = target - np.array(cur)

        u = []
        for i in range(6):
            u.append(diff[i])
            if i <= 2:
                u.append(1)
            else:
                u.append(-1)
        return u

    def cal_leg_ctl(self, env):
        pi = np.pi
        dt = 1. / 240.
        step = env.step_count
        t = step * dt * env.n_time_steps_per_step
        # parameters for alternating tripod
        amp_max = pi / 12
        amps = amp_max * np.ones([3, 6])
        # amps[1,:] = amps[1,:] + pi/8 # boost step height
        # amps[2,:] = amps[1,:] + pi/8 # boost step height
        period = 1
        const_offsets = np.array([[1, 0, -1, 1, 0, -1],
                                  [0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0,
                                   0]]) * np.pi / 8  # offset front and leg base angles a little
        phase_offsets = np.array([
            [0.5, -0.5, 0.5, -0.5, 0.5, -0.5],
            [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]]) * np.pi
        # Create alt tripod leg angles
        joint_pos_command = []
        amps[0, 0:3] = amp_max * np.ones(3) * (-1)
        amps[0, 3:6] = amp_max * np.ones(3) * (1)
        amps[0, :] = np.clip(amps[0, :], -amp_max, amp_max)
        for i in range(6):
            leg_angles_i = amps[:, i] * np.sin(t * 2 * pi / period - phase_offsets[:, i])
            leg_angles_i[1:] = np.clip(leg_angles_i[2:], 0,
                                       np.inf)  # convert up-down motion to up-flat motion
            leg_angles_i += const_offsets[:, i]
            joint_pos_command.append(leg_angles_i)

        #hardcoded for now
        moving_joint_centers = np.asarray([1.570796326794, 0.0, 1.5705]*6)

        joint_pos_command = np.concatenate(joint_pos_command) + moving_joint_centers
        return joint_pos_command

    def step_envs(self, robot_alive,
                  debug_params=None, initial_debug_param_values=None):

        if time_program:
            whole_start = datetime.now()

        envs = self.envs
        terrain_randomizer = self.terrain_randomizer
        num_envs = len(envs)


        # check if debug params have changed
        terrain_changed = False
        terrain_value = None
        if debug_params is not None:
            # for i_param in range(len(debug_params)):
            param_now = envs[-1].p.readUserDebugParameter(debug_params[0])
            if not (param_now == initial_debug_param_values[0]
            ) and terrain_randomizer is not None:
                terrain_randomizer.set_block_heights([envs[0].p], param_now)
                initial_debug_param_values[0] = param_now
                terrain_changed = True
                terrain_value = param_now

        ################# temp hyperparams   #################
        module_to_n_joints = [0, 3, 2, 4]  #n, l, w, b


        if time_program:
            step_start = datetime.now()
        for i_env in range(num_envs):
            if robot_alive[i_env]:
                env = envs[i_env]

                leg_ctl = self.cal_leg_ctl(env)
                wheel_ctl = self.cal_wheel_ctl(env)
                torq_limit = env.moving_joint_max_torques

                #TODO: seperate out the joint indices
                pos_ctl_joint_idx = []
                pos_u = []
                pos_forces = []
                vel_ctl_joint_idx = []
                vel_u = []
                vel_forces = []
                max_vel = []
                cur_joint = 0


                for i in range(len(env.ctl_modules_types)):
                    mod_type = env.ctl_modules_types[i]
                    ctl_len = module_to_n_joints[mod_type]

                    if mod_type == 0: # null
                        pass
                    elif mod_type == 1: #leg
                        pos_ctl_joint_idx = np.concatenate([pos_ctl_joint_idx,
                                                             env.moving_joint_inds[cur_joint:cur_joint+ctl_len]])

                        pos_u = np.concatenate([pos_u, leg_ctl[i*ctl_len:(i+1)*ctl_len]])
                        pos_forces = np.concatenate([pos_forces, torq_limit[cur_joint:cur_joint+ctl_len]])
                        cur_joint += ctl_len
                    elif mod_type == 2: #wheel
                        vel_ctl_joint_idx = np.concatenate([vel_ctl_joint_idx,
                                                            env.moving_joint_inds[cur_joint:cur_joint+ctl_len]])
                        vel_u = np.concatenate([vel_u, wheel_ctl[i*ctl_len:(i+1)*ctl_len]])
                        vel_forces = np.concatenate([vel_forces, torq_limit[cur_joint:cur_joint+ctl_len]])
                        max_vel = np.concatenate([max_vel, env.moving_joint_max_velocities[cur_joint:cur_joint+ctl_len]])
                        cur_joint += ctl_len
                    elif mod_type == 3: #leg-wheel, vel control the wheel, pos control the leg
                        ctl_len = 3
                        pos_ctl_joint_idx = np.concatenate([pos_ctl_joint_idx,
                                                            env.moving_joint_inds[cur_joint:cur_joint + ctl_len]])

                        pos_u = np.concatenate([pos_u, leg_ctl[i * ctl_len:(i + 1) * ctl_len]])
                        pos_forces = np.concatenate([pos_forces, torq_limit[cur_joint:cur_joint + ctl_len]])
                        cur_joint += ctl_len

                        ctl_len = 1
                        vel_ctl_joint_idx = np.concatenate([vel_ctl_joint_idx,
                                                            env.moving_joint_inds[cur_joint:cur_joint + ctl_len]])
                        vel_u = np.concatenate([vel_u, wheel_ctl[i * ctl_len + 1:(i + 1) * ctl_len + 1]])
                        # vel_u = np.concatenate([vel_u, [0]])
                        vel_forces = np.concatenate([vel_forces, torq_limit[cur_joint:cur_joint + ctl_len]])
                        max_vel = np.concatenate(
                            [max_vel, env.moving_joint_max_velocities[cur_joint:cur_joint + ctl_len]])
                        cur_joint += ctl_len

                env.step_mod(pos_u, vel_u, pos_ctl_joint_idx, vel_ctl_joint_idx,
                             pos_forces, vel_forces, max_vel)
                # power = np.sum(np.abs(env.joint_torques * env.joint_vels))
        if time_program:
            print(f"env step takes {datetime.now() - step_start}")
            print(f"whole env step takes {datetime.now() - whole_start}")
        return terrain_changed, terrain_value

    if neural_policy:
        #previous version
        def step_envs(self, robot_alive,
            debug_params = None, initial_debug_param_values=None):

            if time_program:
                whole_start = datetime.now()

            envs = self.envs
            modules = self.modules
            record_video = self.record_video
            max_xy = self.max_xy
            terrain_randomizer = self.terrain_randomizer
            num_envs = len(envs)


            module_action_len= list(np.diff(envs[0].action_indexes))
            attachments = envs[0].attachments
            n_modules = len(envs[0].modules_types)

            env_states = []
            goals_world = []
            for env in envs:
                chassis_yaw = env.pos_rpy[-1]
                chassis_x = env.pos_xyz[0]
                chassis_y = env.pos_xyz[1]

                # set direction to head
                desired_xyyaw = np.zeros(3)
                T= 20
                dt = 20./240.
                speed_scale_xy = (T*dt)*0.314*(1./0.75)
                speed_scale_yaw = (T*dt)*1.1*(1./0.75)

                #TODO: play with it (turn 4 to 3, turn 2 to 0.1)
                desired_xyyaw[0] = speed_scale_xy
                desired_xyyaw[1] = -speed_scale_xy*chassis_y*4
                desired_xyyaw[1] = np.clip(desired_xyyaw[1], -speed_scale_xy, speed_scale_xy)
                # force to turn toward the heading
                desired_xyyaw[2] = -speed_scale_yaw*chassis_yaw/2
                desired_xyyaw[2] = np.clip(desired_xyyaw[2], -speed_scale_yaw,speed_scale_yaw)

                env_state_i = env.get_state()
                env_states.append(env_state_i)

                goals_world.append(torch.tensor(desired_xyyaw,
                        dtype=torch.float32, device=device))

            # check if debug params have changed
            terrain_changed = False
            terrain_value = None
            if debug_params is not None:
                # for i_param in range(len(debug_params)):
                param_now = env.p.readUserDebugParameter(debug_params[0])
                if not(param_now == initial_debug_param_values[0]
                    ) and terrain_randomizer is not None:
                    terrain_randomizer.set_block_heights([envs[0].p],param_now)
                    initial_debug_param_values[0] = param_now
                    terrain_changed = True
                    terrain_value = param_now


            # stack up and pass to gnn in batch
            goals_world = torch.stack(goals_world)
            states = [torch.tensor( np.stack(s),
                             dtype=torch.float32, device=device)
                             for s in list(zip(*env_states)) ]

            node_inputs = create_control_inputs(states, goals_world)

            for module in modules: # this prevents the LSTM in the GNN nodes from
                # learning relations over time, only over internal prop steps.
                module.reset_hidden_states(num_envs)

            if time_program:
                prop_start = datetime.now()

            with torch.no_grad():
                out_mean, out_var = pgnnc.run_propagations(
                    modules, attachments, 2, node_inputs, device)
                u_out_mean = []
                tau_out_mean = []
                for mm in range(n_modules):
                    u_out_mean.append(out_mean[mm][:,:module_action_len[mm]])
                    tau_out_mean.append(out_mean[mm][:,module_action_len[mm]:])
                u_np = torch.cat(u_out_mean,-1).numpy()
                # tau_np = torch.cat(tau_out_mean,-1).numpy()

            if time_program:
                print(f"network prop takes {datetime.now() - prop_start}")

            if time_program:
                step_start = datetime.now()
            for i_env in range(num_envs):
                if robot_alive[i_env]:
                    env = envs[i_env]
                    u = u_np[i_env, :]
                    env.step(u)
                    # power = np.sum(np.abs(env.joint_torques * env.joint_vels))
            if time_program:
                print(f"env step takes {datetime.now() - step_start}")
                print(f"whole env step takes {datetime.now() - whole_start}")
            return terrain_changed, terrain_value