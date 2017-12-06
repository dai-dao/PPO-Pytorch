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


class Trainer(object):
    def __init__(self, env, agent, args):
        self.args = args
        self.env = env
        self.agent = agent

        self.nenv = env.num_envs
        self.obs = np.zeros((self.nenv,) + env.observation_space.shape)
        self.obs = np.transpose(self.obs, (0, 3, 1, 2))
        self.obs[:] = env.reset() # This is channel last
        self.dones = [False for _ in range(self.nenv)]


    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_logpacs = [],[],[],[],[],[]
        epinfos = []        

        for _ in range(self.args.nsteps): # 1 roll-out
            values, actions, logpacs = self.agent.step(self.obs)

            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            mb_logpacs.append(logpacs)
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards) 

        mb_obs = np.asarray(mb_obs)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_logpacs = np.array(mb_logpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)

        last_value, _, _ = self.agent.step(self.obs)

        # discount / boostrap off value
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0   

        print('rewards shape', mb_rewards.size())

        for t in reversed(range(self.args.nsteps)):
            if t == self.args.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_value
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.args.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.args.gamma * self.args.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values

        return (*map(flatten_env_vec, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_logpacs)), epinfos)


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
                    batch_inds = inds[start : end]
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

    print(env.observation_space.shape)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    ppo = PPO_Discrete(env, args)
    trainer = Trainer(env, ppo, args)
    print('Init success')

    trainer.run()
    print('Roll-out success')
    
    # trainer.learn()
    # print('Success')

if __name__ == "__main__":
    test_breakout()