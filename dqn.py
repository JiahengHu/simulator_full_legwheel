import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class dqn(nn.Module):

    def __init__(self, terrain_in_shape,
        n_module_types = 3, 
        max_num_modules = 3, 
        kernel_size= 5, 
        n_channels = 2, 
        n_fc_layers = 4,
        env_vect_size=10,
        hidden_layer_size = 100):
        super(dqn, self).__init__()

        self.terrain_in_shape = terrain_in_shape #(n_channels_in, x, y)
        self.n_module_types = n_module_types
        self.max_num_modules = max_num_modules
        self.kernel_size = kernel_size
        self.n_channels = n_channels # internal channels for conv nets
        self.env_vect_size = env_vect_size
        self.hidden_layer_size = hidden_layer_size
        self.n_fc_layers = n_fc_layers

        # terrain preprocess layers
        self.conv_out_size = terrain_in_shape[2]*terrain_in_shape[1]*n_channels
        self.conv1 = torch.nn.Conv2d(terrain_in_shape[0], n_channels, 
            kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.conv2 = torch.nn.Conv2d(n_channels, n_channels, 
            kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.fc_terrain = nn.Linear(self.conv_out_size, env_vect_size)

        # Combine design and terrain layers
        self.modules_in_size = n_module_types*max_num_modules + max_num_modules
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(
            nn.Linear(env_vect_size+self.modules_in_size, hidden_layer_size))
        for i in range(self.n_fc_layers-2):
            self.fc_layers.append(nn.Linear(hidden_layer_size, hidden_layer_size))
        self.fc_layers.append(nn.Linear(hidden_layer_size, n_module_types))
        # self.fc1 = nn.Linear(env_vect_size+self.modules_in_size, hidden_layer_size)
        # self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        # self.fc3 = nn.Linear(hidden_layer_size, hidden_layer_size)
        # self.fc4 = nn.Linear(hidden_layer_size, n_module_types)


    def forward(self, designs, terrains):
        x = F.relu(self.conv1(terrains))
        x = F.relu(self.conv2(x))
        # print(self.conv_out_size)
        # print(x.shape)
        x = x.view(-1, self.conv_out_size)
        x = F.relu(self.fc_terrain(x))
        # print('forward')
        # print(designs.shape)
        # print(x.shape)

        x = torch.cat( (designs, x), -1)

        for i in range(self.n_fc_layers-1):
            x = F.relu(self.fc_layers[i](x))
        x = self.fc_layers[-1](x)
        
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x


