import torch


class Params(object):
    def __init__(self):
        self.seed = 1 
        self.env_name = 'PongNoFrameskip-v4'
        self.log_dir = './log'

        self.cuda = torch.cuda.is_available()
        self.num_processes = 8
        self.num_stack = 4
        self.lr = 2.5e-4
        self.eps = 1e-5
        self.num_steps = 128 
        self.num_mini_batch = 4
        self.log_interval = 1 
        self.clip_param = 0.1
        self.use_gae = True 
        self.num_frames = 10e6

        self.gamma = 0.99 
        self.tau = 0.95
        self.ppo_epochs = 4
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5


class Breakout_Params(object):
    def __init__(self):
        self.log_interval = 1
        self.cuda = torch.cuda.is_available()
        self.seed = 1
        self.num_stack = 4
        self.lr = 2.5e-4
        self.eps = 1e-5
        self.env_name = 'BreakoutNoFrameskip-v4'
        self.log_dir = './log'
        self.num_processes = 8
        self.clip_param = 0.1
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.gamma = 0.99 
        self.tau = 0.95



        self.value_coefficient = 0.5
        self.entropy_coefficient = 0.01
        self.gamma = 0.99 
        self.lam = 0.95

        self.nsteps = 128
        self.nminibatches = 4
        self.num_update_epochs = 4
        self.lr_schedule = lambda x : x * 2.5e-4
        self.clip_range_schedule = lambda x : x * 0.1
        self.num_timesteps = int(10e6 * 1.1)

        self.epsilon_min = 0.1 
        self.annealing_end = 1000000.