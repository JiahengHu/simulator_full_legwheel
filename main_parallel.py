'''

TODO 
- save file with all recorded data so far, so that later I can 
play it back faster and see if I can improve training speed
- during eval print out not just the one that just evaled,
but all possible state action values on the tree, recursively called,
( can be used to get the probability of random selection of each?)

'''



import torch
# from replay_buffer import replay_buffer
from replay_buffer_tensors import replay_buffer
from design_assembler import module_types, num_module_types, module_penalties
from design_assembler import add_module, module_vector_list_to_robot_name
from dqn import dqn
import os
import logging
import numpy as np
import time
from simulation_runner import simulation_runner, terrain_grid_shape, reward_function
import traceback

# utility that helps manage the logging mess for multiple workers
# downloaded the main file from
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
print = logging.info

print('module_penalties: ' + str(module_penalties))

### hyperparameters
cpu_count = torch.multiprocessing.cpu_count()
if cpu_count > 20:
    NUM_SIM_WORKERS = 16
elif cpu_count > 10:
    NUM_SIM_WORKERS = 8
elif cpu_count > 5:
    NUM_SIM_WORKERS = 5
else:
    NUM_SIM_WORKERS = 1

SAVE_BUFFER = True # flag to keep some statistics on the rewards obtained by simulation
REPLAY_MEMORY_SIZE = 5000
LR_INIT = 1e-4
N_ACTIONS = num_module_types
MAX_N_MODULES = 3
NUM_ENVS = 3 # number of environments to run in parallel
SIM_TIME_STEPS = 250 # determines the farthest possible travel distance
NUM_EPISODES = 20000
TARGET_UPDATE = 100 # how many episodes to delay target net from policy net
BATCH_SIZE = 200 # number of samples in a batch for dqn learning
BOLTZMANN_TEMP_START = 10 
BOLTZMANN_TEMP_MIN = 2
BOLTZMANN_TEMP_DECAY_CONST = 1./4000 # T = T0*exp(-c*episode) e.g. 10*np.exp(-np.array([0, 1000, 5000])/1000)
# RELOAD_WEIGHTS=True # if true, looks for policy_net_weights.pt to load in
RELOAD_WEIGHTS=False # if true, looks for policy_net_weights.pt to load in
SAVE_EP = 100 # how many episodes to wait between saving
VALIDATION_EP = 100 # how many episdoes to wait between validations


# For testing only
# BATCH_SIZE = 10 # number of samples in a batch for dqn learning
# VALIDATION_EP = 10 # how many episdoes to wait between validations

# temperature for boltzmann exploration 
def boltzmann_temp(episode):
    return max(BOLTZMANN_TEMP_START*np.exp(-episode*BOLTZMANN_TEMP_DECAY_CONST),
                BOLTZMANN_TEMP_MIN)

# select an action randomly with boltzmann exploration
def select_boltzmann_action(policy_net, designs, terrains, boltzmann_temp=1):
    with torch.no_grad():
        actions_out = policy_net(designs, terrains)
        # Scale by a temperature factor. as temperature gets lower, tends towards uniform outputs. as higher, tends towards true max. 
        actions_softmax = torch.nn.functional.softmax(actions_out/boltzmann_temp, dim=-1)
    
    # TODO: replace with torch categorical 
    action_inds = []
    for ind in range(actions_softmax.shape[0]):
        chosen = np.random.uniform(0, 1)
        cumulative = 0
        for action_index in range(N_ACTIONS):
            cumulative += actions_softmax[0][action_index].item() # [0] bc first dimension is blank for stacking
            if cumulative >= chosen:
                action_inds.append(action_index)
                break 
    return torch.tensor(action_inds, dtype=torch.long), actions_softmax

def select_epsgreedy_action(policy_net, designs, terrains, eps):
    with torch.no_grad():
        actions_out = policy_net(designs, terrains)
    if np.random.rand()>eps:
        action_inds = actions_out.max(1)[1].cpu()
    else:
        action_inds = torch.randint(0, N_ACTIONS, (designs.shape[0],))
    return action_inds, actions_out 

# selects an action with the max Q value from all actions.
def select_max_action(policy_net, designs, terrains):
    with torch.no_grad():
        actions_out = policy_net(designs, terrains)
        action_inds = actions_out.max(1)[1]
        return action_inds, actions_out 

# def run_episode_test(policy_net,replay_memory):
#     terr_i = torch.zeros(policy_net.terrain_in_shape)
#     des_i = torch.zeros(1, policy_net.n_module_types*policy_net.max_num_modules + policy_net.max_num_modules)
#     next_des_i = torch.zeros(1, policy_net.n_module_types*policy_net.max_num_modules + policy_net.max_num_modules)
#     action = torch.zeros(1, policy_net.n_module_types)
#     reward = torch.zeros(1,1)
#     non_final_i = torch.zeros(1,1)
#     replay_memory.push(des_i, terr_i, action, 
#                         next_des_i, reward, non_final_i)


def run_episode(policy_net,
            pipe,
            is_training_episode,
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
        for i_env in range(n_designs):
            next_designs[i_env,:], penalty = add_module(
                                    designs[i_env,:], 
                                    i_dqn, MAX_N_MODULES,
                                    actions[i_env])

        reward = -torch.tensor(penalty,dtype=torch.float32) # adding a module has no cost for now
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
                sim_runner.load_robots(robot_name)
                rewards = sim_runner.run_sims(n_time_steps=SIM_TIME_STEPS)
                # if (sim_runner.reward_function == 'Testing Proxy' or 
                #     sim_runner.reward_function == 'Recorded Simulation'):
                #     time.sleep(0.1)

                reward += rewards.mean()
                if sim_runner.is_valid:
                    # print(terrains)
                    terrain_max = terrains.max().numpy()
                    print(print_str + ' simulated ' + str(robot_name) +
                        ' rewards ' +
                            np.array2string(rewards.numpy(),precision=1) 
                            + ' Terrain max ' + np.array2string(terrain_max,precision=3) )
            else:
                print(robot_name)

        else:
            non_final = torch.tensor(1, dtype=torch.bool)
        

        if is_training_episode:
            # add to replay buffer
            for i_env in range(n_designs):
                # action = actions[i_env].unsqueeze(0).clone()
                # des_i = designs[i_env].clone()
                # terr_i = terrains[i_env].clone()
                # next_des_i = next_designs[i_env].clone()
                # reward = reward.squeeze().clone()
                # non_final_i  = non_final.clone()
                # action = actions[i_env].unsqueeze(0).numpy()
                action = actions[i_env].numpy()
                des_i = designs[i_env].numpy()
                terr_i = terrains[i_env].numpy()
                next_des_i = next_designs[i_env].numpy()
                reward = reward.squeeze().numpy()
                non_final_i  = non_final.numpy()
                # try:
                pipe.send([des_i, terr_i, action, 
                        next_des_i, reward, non_final_i])

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


        # select randomized terrain for training episode
        terrain = sim_runner.randomize_terrains()
        current_boltzmann_temp = boltzmann_temp(i_episode)  # anneals temp
        print_str_now = 'worker:' + str(worker_num) + ', ep:' + str(i_episode)
        run_episode(policy_net, pipe, True,
             terrain, sim_runner,
              current_boltzmann_temp =  current_boltzmann_temp,
              print_str = print_str_now )
        # run_episode_test(policy_net,replay_memory)


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

                run_episode(policy_net,pipe,
                    False,terrain) # for validation, don't simulate or store anything,
                # run with a range of terrains to check output

                print('-----------')



if __name__== "__main__":

    # spawn processes
    torch.multiprocessing.set_start_method('spawn') # needed for CUDA drivers in parallel
    torch.multiprocessing.set_sharing_strategy('file_system')

    # device = torch.device('cpu')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    
    ### Initialize and load policy
    # initiate the policy network 
    PATH = os.path.join(cwd, 'policy_net_params.pt')

    if RELOAD_WEIGHTS and os.path.exists(PATH):
        save_dict = torch.load(PATH)
        save_dict = dict()

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
    else:
        print('Creating ' + PATH)
        policy_net = dqn(terrain_grid_shape, 
                     max_num_modules = MAX_N_MODULES,
                     n_conv_layers = 2,
                     hidden_layer_size = 150)

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

    policy_net_copy = dqn( 
        terrain_in_shape = policy_net.terrain_in_shape ,
        n_module_types= policy_net.n_module_types,
        max_num_modules=policy_net.max_num_modules,
        kernel_size=policy_net.kernel_size,
        n_channels= policy_net.n_channels,
        n_fc_layers=policy_net.n_fc_layers,
        env_vect_size=policy_net.env_vect_size,
        hidden_layer_size=policy_net.hidden_layer_size,
        n_conv_layers = policy_net.n_conv_layers).to(device)

    target_net = dqn( 
        terrain_in_shape = policy_net.terrain_in_shape ,
        n_module_types= policy_net.n_module_types,
        max_num_modules=policy_net.max_num_modules,
        kernel_size=policy_net.kernel_size,
        n_channels= policy_net.n_channels,
        n_fc_layers=policy_net.n_fc_layers,
        env_vect_size=policy_net.env_vect_size,
        hidden_layer_size=policy_net.hidden_layer_size,
        n_conv_layers = policy_net.n_conv_layers).to(device)


    optimizer = torch.optim.Adam(policy_net_copy.parameters(),
                   lr=LR_INIT, weight_decay= 1e-4)


    policy_net_copy.load_state_dict(policy_net.state_dict())
    target_net.load_state_dict(policy_net_copy.state_dict())
    target_net.eval()
    opt_ep = 0
    while current_episode.value < NUM_EPISODES:

        # gather data from the pipes
        alive_list= []
        for i in range(num_p):
            process = processes[i]
            i_alive = process.is_alive()
            alive_list.append(i_alive)
            if i_alive:
                while pipes[i].poll():
                    # print('Pipe fileno: ' + str(pipes[i].fileno()))
                    pipe_read = pipes[i].recv()
                    # replay_memory.push(pipe_read[0], pipe_read[1], 
                    #     pipe_read[2], pipe_read[3], pipe_read[4], pipe_read[5] )
                    replay_memory.push(torch.tensor(pipe_read[0]),
                                torch.tensor(pipe_read[1]), 
                                torch.tensor(pipe_read[2]), 
                                torch.tensor(pipe_read[3]),
                                torch.tensor(pipe_read[4]),
                                torch.tensor(pipe_read[5]) )
                    # Note: it seems to be better to pass numpy over the pipe
                    # rather than 
                    # print('pushed to memory ' + str(pipe_read[0].data[0:3])) 
                    # print('pushed to memory ')
        if not(np.any(alive_list)):
            print('all workers ended')
            break

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
            # loss = torch.nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values)
            # compute mean squared error loss
            loss = torch.nn.functional.mse_loss(state_action_values,
                              expected_state_action_values)


            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            # clamp the gradients
            for param in policy_net_copy.parameters():
                if param.grad is not None:
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
                    if param_group['lr'] >= 2e-6:  # set min learning rate
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
                save_dict['n_conv_layers'] = policy_net.n_conv_layers
                save_dict['i_episode'] = i_episode
                save_dict['module_penalties'] = module_penalties
                torch.save(save_dict, PATH)

                if SAVE_BUFFER:
                    PATH2 = os.path.join(cwd, 'buffer_memory.pt')
                    save_dict2 = replay_memory.get_dict()
                    save_dict2['reward_function'] = reward_function
                    torch.save(save_dict2, PATH2)
        time.sleep(0.01) # keep loop from being too fast



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

    print('Done training')


'''
NOTES
"RuntimeError: received 0 items of ancdata"
might be related to ulimit -n



Traceback (most recent call last):
  File "main_parallel.py", line 403, in <module>
    pipe_read[2], pipe_read[3], pipe_read[4], pipe_read[5] )
  File "/home/cobracommander/anaconda3/lib/python3.7/multiprocessing/connection.py", line 251, in recv
    return _ForkingPickler.loads(buf.getbuffer())
  File "/home/cobracommander/anaconda3/lib/python3.7/site-packages/torch/multiprocessing/reductions.py", line 276, in rebuild_storage_fd
    fd = df.detach()
  File "/home/cobracommander/anaconda3/lib/python3.7/multiprocessing/resource_sharer.py", line 58, in detach
    return reduction.recv_handle(conn)
  File "/home/cobracommander/anaconda3/lib/python3.7/multiprocessing/reduction.py", line 185, in recv_handle
    return recvfds(s, 1)[0]
  File "/home/cobracommander/anaconda3/lib/python3.7/multiprocessing/reduction.py", line 161, in recvfds
    len(ancdata))
RuntimeError: received 0 items of ancdata




'''