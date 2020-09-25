'''
functions that step forward from an empty robot to a complete robot,
by adding one part at a time



'''
import numpy as np
import torch

# modules:
# 0 = no module
# 1 = leg
# 2 = wheel
module_types = ['n', 'l', 'w']
num_module_types = len(module_types)


def num_list_to_one_hot(num_list):
    module_vector_list = []
    for n in num_list:
        v = torch.zeros(num_module_types) 
        v[n] = 1
        module_vector_list.append(v)
    return module_vector_list

def module_vector_list_to_robot_name(module_vector_list):
    letters = ''
    for mv in module_vector_list:
        nonzero_inds=torch.nonzero(mv)
        # i for invalid because it is empty
        if len(nonzero_inds)>0:
            letters += module_types[nonzero_inds[0]]
        else:
            letters += 'i'
    # double and flip for symmetric design
    robot_name = letters #+ letters[::-1]
    return robot_name

def add_module(design_in, current_module, max_n_modules, action):
    design_out = design_in.clone()
    # print(num_module_types*current_module + action)
    design_out[ num_module_types*current_module + action] = 1
    design_out[ num_module_types*num_module_types:] = 0
    if current_module < (max_n_modules-1):
        design_out[ num_module_types*num_module_types + current_module+1] = 1

    return design_out


if __name__== "__main__":

    num_list = [1, 0 , 2]  
    module_vector_list = num_list_to_one_hot(num_list)
    robot_name = module_vector_list_to_robot_name(module_vector_list)
    print(robot_name)


