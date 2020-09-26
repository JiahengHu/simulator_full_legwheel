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
# from simulation_runner_batch import simulation_runner
from simulation_runner import simulation_runner
from dqn import dqn
import torch.nn.functional as F
import random
import os
import logging
import numpy as np


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
print = logging.info

# device = torch.device('cpu')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

### hyperparameters
RECORD_REWARDS = False # flag to keep some statistics on the rewards obtained by simulation
REPLAY_MEMORY_SIZE = 5000
LR_INIT = 1e-4
N_ACTIONS = num_module_types
MAX_N_MODULES = 3
NUM_ENVS = 3 # number of environments to run in parallel
SIM_TIME_STEPS = 100
NUM_EPISODES = 10000
TARGET_UPDATE = 50 # how many episodes to delay target net from policy net
BATCH_SIZE = 200 # number of samples in a batch for dqn learning
# BATCH_SIZE = 10 # number of samples in a batch for dqn learning
N_TRAIN_ITERS = 10 # number of training steps after each sim batch
BOLTZMANN_TEMP_START = 10 
BOLTZMANN_TEMP_MIN = 0.5
BOLTZMANN_TEMP_DECAY_CONST = 1./2000 # T = T0*exp(-c*episode) e.g. 10*np.exp(-np.array([0, 1000, 5000])/1000)
# RELOAD_WEIGHTS=True # if true, looks for policy_net_weights.pt to load in
RELOAD_WEIGHTS=False # if true, looks for policy_net_weights.pt to load in


### Initialize and load policy
sim_runner = simulation_runner(NUM_ENVS)
terrain_grid_shape = sim_runner.terrain_grid_shape
### Initialize DQN and replay buffer

# initiate the policy network and target network. 

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
        hidden_layer_size=save_dict['hidden_layer_size']).to(device)
    target_net = dqn( 
        terrain_in_shape = save_dict['terrain_in_shape'] ,
        n_module_types= save_dict['n_module_types'] ,
        max_num_modules=save_dict['max_num_modules'] ,
        kernel_size=save_dict['kernel_size'],
        n_channels= save_dict['n_channels'],
        n_fc_layers=save_dict['n_fc_layers'],
        env_vect_size=save_dict['env_vect_size'],
        hidden_layer_size=save_dict['hidden_layer_size']).to(device)
    policy_net.load_state_dict( save_dict['policy_net_state_dict'])
    print('Reloaded weights from ' + PATH)
else:
    print('Creating ' + PATH)
    policy_net = dqn(terrain_grid_shape, 
                 max_num_modules = MAX_N_MODULES).to(device)
    target_net = dqn(terrain_grid_shape, 
                 max_num_modules = MAX_N_MODULES).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = torch.optim.Adam(policy_net.parameters(),
                   lr=LR_INIT, weight_decay= 1e-4)

reward_record_dict = dict()

# replay buffer
replay_memory = replay_buffer(REPLAY_MEMORY_SIZE)


# temperature for boltzmann exploration 
def boltzmann_temp(episode):
    return max(BOLTZMANN_TEMP_START*np.exp(-episode*BOLTZMANN_TEMP_DECAY_CONST),
                BOLTZMANN_TEMP_MIN)

# select an action randomly with boltzmann exploration
def select_boltzmann_action(designs, terrains, boltzmann_temp=1):
    with torch.no_grad():
        actions_out = policy_net(designs, terrains)
        # Scale by a temperature factor. as temperature gets lower, tends towards uniform outputs. as higher, tends towards true max. 
        actions_softmax = F.softmax(actions_out/boltzmann_temp, dim=-1)
        
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

def select_epsgreedy_action(designs, terrains, eps):
    with torch.no_grad():
        actions_out = policy_net(designs, terrains)

    if np.random.rand()>eps:
        action_inds = actions_out.max(1)[1].cpu()

    else:
        action_inds = torch.randint(0, N_ACTIONS, (designs.shape[0],))

    return action_inds, actions_out 

# selects an action with the max Q value from all actions.
def select_max_action(designs, terrains):
    with torch.no_grad():
        actions_out = policy_net(designs, terrains)
        action_inds = actions_out.max(1)[1]
        return action_inds, actions_out 

def run_episode(is_training_episode, terrains, 
                sim_runner_now=None, 
                current_boltzmann_temp= 5 ):

    # if not(is_training_episode):


    # initialize empty robot batch
    designs = torch.zeros(1, N_ACTIONS*MAX_N_MODULES + MAX_N_MODULES)
    designs[:, N_ACTIONS*MAX_N_MODULES] = 1 # indicate which port adding now
    
    # loop dqn until done:
    for i_dqn in range(MAX_N_MODULES):

        # run through generator to get actions
        if is_training_episode:
            actions, actions_softmax = select_boltzmann_action(designs.to(device),
                             terrains.to(device), current_boltzmann_temp)
            # actions, actions_softmax = select_epsgreedy_action(designs.to(device),
                             # terrains.to(device), 0.9)
        else:
            actions, state_action_values = select_max_action(designs.to(device),
                             terrains.to(device))

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
                sim_runner_now.load_robots(robot_name)
                rewards = sim_runner_now.run_sims()
                reward += rewards.mean()
                if sim_runner_now.is_valid:
                    print('Ran simulations of ' + str(robot_name) +
                        ' rewards ' +
                            np.array2string(
                            rewards.numpy(),precision=1))

                if RECORD_REWARDS:
                    # keep a running list of data seen so far 
                    # MIGHT BLOW UP MEMORY- might need to keep mean and std deve instead
                    for i_env in range(NUM_ENVS):
                        key = robot_name
                        if key not in reward_record_dict:
                            reward_record_dict[key] = list()
                        reward_now = rewards[i_env].item()
                        # leave the invalid entries as empty to show they have been visited and 
                        # deemed invalid
                        if reward_now>-9:
                            reward_record_dict[key].append(reward_now)



        else:
            non_final = torch.tensor(1, dtype=torch.bool)
        

        if is_training_episode:
            # add to replay buffer
            for i_env in range(1):
                action = actions[i_env].unsqueeze(0).clone()
                # reward = reward.unsqueeze(0).clone()
                reward = reward.clone()
                replay_memory.push(designs[i_env].clone(), terrains[i_env].clone(),
                    action, next_designs[i_env].clone(), reward, non_final.clone())
                #('state', 'terrain', 'action', 'next_state', 'reward', 'done'))
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

    if not is_training_episode and RECORD_REWARDS:
        print('Mean rewards seen so far:')
        for key in reward_record_dict:
            design_mean = np.mean(reward_record_dict[key])
            if design_mean>-9: # only print those that are valid
                print(key + ': ' + str(design_mean))

        print('-------------')



    if not(is_training_episode):
        # terrain_means = []
            # terrain_means.append(torch.mean(terrains[i_env]).numpy().item())
        terrain_max=torch.max(terrains).numpy().item()
        # print('terrain means: ' + str(terrain_means))
        print('terrain max: ' + str(terrain_max))
        print('robot_name: ' + str(robot_name))

# For outer iteration:
for i_episode in range(NUM_EPISODES):

    # select randomized batch of terrains
    terrain = sim_runner.randomize_terrains()
    temp = boltzmann_temp(i_episode)  # anneals temp
    run_episode(True,terrain,sim_runner,
          current_boltzmann_temp =  temp)
       

    ### run dqn training cycles
    # for training cycles:
        # sample from buffer
    if len(replay_memory) >= BATCH_SIZE:
        for i_train in range(N_TRAIN_ITERS):    
            batch = replay_memory.sample(BATCH_SIZE)

            # Compute a mask of non-final states and concatenate the batch elements
            # (a final state would've been the one after which simulation ended)
            non_final_mask = torch.stack(batch.non_final)
            # indexing with dtype torch.uint8 is deprecated, cast to dtype torch.bool

            state_batch = torch.stack(batch.state).to(device)
            next_state_batch = torch.stack(batch.next_state).to(device)
            terrain_batch = torch.stack(batch.terrain).to(device)
            action_batch = torch.stack(batch.action).to(device)
            reward_batch = torch.cat(batch.reward).to(device)
            non_final_next_states = next_state_batch[non_final_mask]
            non_final_terrains = terrain_batch[non_final_mask]

            # forward pass with autograd
            # Compute Q(s_t, a)
            state_action_values_raw = policy_net(
                                     state_batch,terrain_batch)
            state_action_values = state_action_values_raw.gather(
                                      1, action_batch).squeeze()

            # Compute V(s_{t+1}) for next states.
            next_state_values = torch.zeros(BATCH_SIZE, device=device)
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
            for param in policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()


        if (i_episode) % 10==0 and i_episode>0:

            # print('------------')
            # print('state_batch')
            # print(state_batch)
            # print('action_batch')
            # print(action_batch)
            # print('next_state_batch')
            # print(next_state_batch)
            # print('reward_batch')
            # print(reward_batch)
            # print('non_final_mask')
            # print(non_final_mask)
            # print('state_action_values_raw')
            # print(state_action_values_raw)
            # print('state_action_values')
            # print(state_action_values)
            # print('expected_state_action_values')
            # print(expected_state_action_values)

            print('Loss at ep. ' + str(i_episode) + ': ' + str(loss.detach().cpu().numpy()))


    # Update the target network, copying all weights and biases in DQN
    if (i_episode) % TARGET_UPDATE == 0 and i_episode>0:
        target_net.load_state_dict(policy_net.state_dict())


    # if (i_episode % 20 == 0):
    if (i_episode % 100 == 0 and i_episode>0):
        print('Boltzmann temp at ep ' + str(i_episode) + ': ' + str(temp))
        print('--- eval at ep ' + str(i_episode) + ' ---')
        for terrain_block_height in np.linspace(
                sim_runner.MAX_BLOCK_HEIGHT_LOW,
                sim_runner.MAX_BLOCK_HEIGHT_HIGH, 3):
            terrain = sim_runner.randomize_terrains(
                terrain_block_height=terrain_block_height)
            run_episode(False,terrain) # for validation, don't simulate or store anything,
            # run with a range of terrains to check output

            # compare with a range of real robots:
            test_robot_list = ['lll', 'lwl', 'wnw']
            test_robot_rewards = []
            out_str ='Test rewards: '
            for test_robot_name in test_robot_list:
                sim_runner.load_robots(test_robot_name)
                test_robot_rewards.append(sim_runner.run_sims())
                out_str += np.array2string(
                    test_robot_rewards[-1].numpy(),precision=1)
            print('Test robots:' + str(test_robot_list))
            print( out_str )

            print('-----------')

    if (i_episode % 100 == 0):
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

        torch.save(save_dict, PATH)
        # PATH2 = os.path.join(cwd, 'rewards_from_simulation.pt')
        # torch.save(reward_record_dict, PATH2)

print('Done training')
# Done training








