import gym 
import numpy as np 
import time
from params import Breakout_Params, Params
from envs import *
from ppo import *
from collections import deque
import os.path as osp
from utils import * 

from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines import bench, logger

import torch


class TrainerPlus(object):
    def __init__(self, env, agent, args):
        self.args = args
        self.env = env
        self.agent = agent

        self.dtype = torch.FloatTensor 
        self.atype = torch.LongTensor
        if args.cuda:
            self.dtype = torch.cuda.FloatTensor
            self.atype = torch.cuda.LongTensor

        self.nenv = env.num_envs
        # self.obs = np.zeros((self.nenv,) + env.observation_space.shape)

        self.obs = torch.from_numpy(env.reset()).type(self.dtype) # This is channel first
        self.dones = torch.zeros(self.nenv).type(self.dtype)

        self.mb_obs = torch.zeros(self.args.nsteps, *self.obs.size()).type(self.dtype)
        self.mb_rewards = torch.zeros(self.args.nsteps, self.nenv).type(self.dtype)
        self.mb_actions = torch.zeros(self.args.nsteps, self.nenv).type(self.atype)
        self.mb_values = torch.zeros(self.args.nsteps, self.nenv).type(self.dtype)
        self.mb_dones = torch.zeros(self.args.nsteps, self.nenv).type(self.dtype)
        self.mb_logpacs = torch.zeros(self.args.nsteps, self.nenv).type(self.dtype)
        self.mb_returns = torch.zeros(self.args.nsteps, self.nenv).type(self.dtype)
        self.mb_advs = torch.zeros(self.args.nsteps, self.nenv).type(self.dtype)


    def run(self):
        epinfos = []        

        for step in range(self.args.nsteps): # 1 roll-out
            values, actions, logpacs = self.agent.step(self.obs)
            cpu_actions = actions.data.cpu().numpy().astype(np.int32).reshape((-1))

            self.mb_obs[step].copy_(self.obs)
            self.mb_values[step].copy_(values.data.view(-1))
            self.mb_actions[step].copy_(actions.data.view(-1))
            self.mb_dones[step].copy_(self.dones.view(-1))
            self.mb_logpacs[step].copy_(logpacs.data.view(-1))

            obs, rewards, dones, infos = self.env.step(cpu_actions)
            self.obs.copy_(torch.from_numpy(obs).type(self.dtype))
            self.dones.copy_(torch.from_numpy(dones.astype(int)).type(self.dtype))

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            self.mb_rewards[step].copy_(torch.from_numpy(rewards).type(self.dtype)) 

        last_value, _, _ = self.agent.step(self.obs)
        last_value = last_value.data.view(-1)

        # discount / boostrap off value
        lastgaelam = 0   

        for t in reversed(range(self.args.nsteps)):
            if t == self.args.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_value
            else:
                nextnonterminal = 1.0 - self.mb_dones[t+1]
                nextvalues = self.mb_values[t+1]
            delta = self.mb_rewards[t] + self.args.gamma * nextvalues * nextnonterminal - self.mb_values[t]
            lastgaelam = delta + self.args.gamma * self.args.lam * nextnonterminal * lastgaelam
            self.mb_advs[t].copy_(lastgaelam)
        self.mb_returns.copy_(self.mb_advs + self.mb_values)
        return (*map(flatten_env_vec, (self.mb_obs, self.mb_returns, self.mb_dones, self.mb_actions, self.mb_values, self.mb_logpacs)), epinfos)


    def learn(self):
        # Number of samples in one roll-out
        nbatch = self.nenv * self.args.nsteps
        nbatch_train = nbatch // self.args.nminibatches

        # Total number of steps to run simulation
        total_timesteps = self.args.num_timesteps
        # Number of times to run optimization
        nupdates = int(total_timesteps // nbatch)

        epinfobuf = deque(maxlen=100)

        for update in range(1, nupdates+1):
            assert nbatch % self.args.nminibatches == 0

            # Adaptive clip-range and learning-rate decaying
            frac = 1.0 - (update - 1.0) / nupdates
            lrnow = self.args.lr_schedule(frac)
            cliprangenow = self.args.clip_range_schedule(frac)
            num_steps_so_far = update * nbatch

            before_run = time.time()
            obs, returns, masks, actions, values, logpacs, epinfos = self.run()
            run_time = time.time() - before_run

            epinfobuf.extend(epinfos)
            inds = np.arange(nbatch)
            mblossvals = []

            before_update = time.time()
            for _ in range(self.args.num_update_epochs):
                np.random.shuffle(inds)
                # Per mini-batches in one roll-out
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    batch_inds = torch.from_numpy(inds[start : end]).type(self.atype)
                    slices = (arr[batch_inds] for arr in (obs, returns, masks, actions, values, logpacs))
                    pg_loss, vf_loss, entropy = self.agent.update(*slices, lrnow, cliprangenow)
                    mblossvals.append([pg_loss, vf_loss, entropy])
            update_time = time.time() - before_update

            # Logging
            lossvals = np.mean(mblossvals, axis=0)

            if update % self.args.log_interval == 0 or update == 1:
                logger.logkv("Run time", run_time)
                logger.logkv("Update time", update_time)
                logger.logkv("serial_timestep", update * self.args.nsteps)
                logger.logkv("num_updates", update)
                logger.logkv("total_timesteps", update * nbatch)
                logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
                logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
                for (lossval, lossname) in zip(lossvals, self.agent.loss_names):
                    logger.logkv(lossname, lossval)
                logger.dumpkvs()

        self.env.close()



def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def test_breakout():
    logger.configure('./log', ['stdout', 'tensorboard'])
    args = Breakout_Params()

    
    nenvs = 8
    env = SubprocVecEnv([make_env(i, 'BreakoutNoFrameskip-v4') for i in range(nenvs)])
    env = PyTorch_VecFrameStack(env, args.num_stack)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    ppo = PPO_Discrete(env, args)
    trainer = TrainerPlus(env, ppo, args)
    print('Init success')

    # trainer.run()
    # print('Roll-out success')
    
    trainer.learn()
    print('Success')

if __name__ == "__main__":
    test_breakout()

