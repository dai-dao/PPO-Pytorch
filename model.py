import torch 
import torch.nn as nn
import torch.nn.functional as F 
from utils import orthogonal 
from distributions import Categorical
import numpy as np 


def weight_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        orthogonal(m.weight.data)
    
        if m.bias is not None:
            m.bias.data.fill_(0)


class FFPolicy(nn.Module):
    def __init__(self):
        super(FFPolicy, self).__init__()
        self.dist = None


    def forward(self, inputs, states, masks):
        raise NotImplementedError


    def act(self, inputs, deterministic=False):
        value, x = self(inputs)
        action = self.dist.sample(x, deterministic=deterministic)
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, action)
        return value, action, action_log_probs


    def evaluate_actions(self, inputs, actions):
        value, x = self(inputs)
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, actions)
        return value, action_log_probs, dist_entropy


class CNNPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space):
        super(CNNPolicy, self).__init__()

        self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)
        self.linear1 = nn.Linear(32 * 7 * 7, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.dist = Categorical(512, action_space.n)

        self.train()
        self.reset_parameters()
    

    @property
    def state_size(self):
        if hasattr(self, 'gru'):
            return 512
        else:
            return 1


    def reset_parameters(self):
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)


    def forward(self, inputs):
        x = F.relu(self.conv1(inputs / 255.))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.linear1(x))

        return self.critic_linear(x), x

        