import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np 
import time 
from model import *
import torch.optim as optim 
from storage import RolloutStorage
import torch



class PPO_Discrete(object):
    def __init__(self, env, args):
        self.env = env
        self.args = args 
        self.obs_shape = self.env.observation_space.shape
        self.net = CNNPolicy(4, self.env.action_space)
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy']

        if self.args.cuda:
            self.net.cuda()
        
        self.optimizer = optim.Adam(self.net.parameters(), self.args.lr, eps=self.args.eps)
    

    def step(self, s):
        # s = np.transpose(s, (0, 3, 1, 2))
        # s = Variable(torch.from_numpy(s).type(self.T), volatile=True)
        s = Variable(s, volatile=True)
        value, action, action_log_prob = self.net.act(s)
        # cpu_actions = action.data.cpu().numpy().astype(np.int32).reshape((-1))
        # value = value.data.cpu().numpy().reshape((-1))
        # action_log_prob = action_log_prob.data.cpu().numpy().reshape((-1))
        # return value, cpu_actions, action_log_prob
        return value, action, action_log_prob


    def update(self, obs, returns, masks, actions, values, neglogpacs, lrnow, cliprange_now):
        advantages = returns - values 
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        obs = Variable(obs)
        actions = Variable(actions).view(-1, 1)
        values = Variable(values).view(-1, 1)
        returns = Variable(returns).view(-1, 1)
        oldpi_prob = Variable(neglogpacs).view(-1, 1)
        advantages = Variable(advantages).view(-1, 1)

        vpred, action_log_probs, dist_entropy = self.net.evaluate_actions(obs, actions)
        ratio = torch.exp(action_log_probs - oldpi_prob)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - cliprange_now, 1.0 + cliprange_now) * advantages
        action_loss = -torch.min(surr1, surr2).mean()
        
        value_loss = (returns - vpred).pow(2).mean()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lrnow

        self.optimizer.zero_grad()
        (value_loss + action_loss - dist_entropy * self.args.entropy_coef).backward()
        nn.utils.clip_grad_norm(self.net.parameters(), self.args.max_grad_norm)
        self.optimizer.step()

        return action_loss.data.cpu().numpy()[0], value_loss.data.cpu().numpy()[0], \
                        dist_entropy.data.cpu().numpy()[0]



