import torch
import torch.nn as nn
import torch.nn.functional as F


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()   
        self.linear = nn.Linear(num_inputs, num_outputs)


    def forward(self, x):
        x = self.linear(x)
        return x 


    def sample(self, x, deterministic):
        x = self(x)
        probs = F.softmax(x)

        if deterministic is False:
            action = probs.multinomial()
        else:
            action = probs.max(1, keepdim=True)[1]

        return action 

    
    def logprobs_and_entropy(self, x, action):
        x = self(x)
        log_probs = F.log_softmax(x)
        probs = F.softmax(x)

        action_log_probs = log_probs.gather(1, action)
        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return action_log_probs, dist_entropy

