# Based on
# https://github.com/ikostrikov/pytorch-a3c
import argparse
import os
import gymnasium

import torch
import torch.multiprocessing as mp

from torch.multiprocessing import Manager

from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.atari_wrappers import MaxAndSkipEnv

from envs import create_atari_env

import my_optim

from model import ActorCritic
from test import test
from train import train

import sys
sys.path.append("../../game")

import random

# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--gae-lambda', type=float, default=1.00,
                    help='lambda parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=4,
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=1000000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--env-name', default='breakwall_clone:breakwall',
                    help='environment to train on (default: breakwall_clone:breakwall)')
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')
parser.add_argument('--build-name')
parser.add_argument('--eval-mode', action='store_true')
parser.add_argument('--eval-len', type=int, default=30)


def have_model_weights(build):
    path = "checkpoints/" + build + ".pt"
    return os.path.exists(path)

def load_model_weights(model, build):
    path = "checkpoints/" + build + ".pt"
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))

def randomize_env_args():
    arguments = {"vel_inc": [1.001, 1.0025, 1.005, 1.01, 1.02],
                 "max_spd": [1.5, 2.0, 2.5],
                 "paddle_spd": [3.0, 3.5, 4.0],
                 "offset_row": [5, 10, 15, 20],
                 "brick_color": [[66, 72, 200], [100, 0, 100], [200, 72, 66]]}
    
    res = {}
    for arg, values in arguments.items():
        res[arg] = random.choice(values)
    return res

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    args = parser.parse_args()

    mp.set_start_method('spawn')

    if args.eval_mode:
        env_arguments = randomize_env_args()
        print("env arguments: ", env_arguments)
    else:
        env_arguments = {}

    torch.manual_seed(args.seed)
    env_name = args.env_name

    manager = Manager()

    # Use the Baseline Atari environment because of Deepmind helper functions
    env = create_atari_env(env_name, env_arguments)

    #env.seed(seed)
    
    have_wts = have_model_weights(args.build_name)
    shared_model = ActorCritic(
        env.observation_space.shape[0], env.action_space, not have_wts)
    load_model_weights(shared_model, args.build_name)
    shared_model.share_memory()

    if args.no_shared:
        optimizer = None
    else:
        optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

    processes = []

    counter = mp.Value('i', 0)
    finish_signal = mp.Value('i', 0)
    mp_rewards = manager.list()
    lock = mp.Lock()

    p = mp.Process(target=test, args=(args.num_processes, args, shared_model, counter, lock, mp_rewards, not have_wts, env_arguments, finish_signal))
    p.start()
    processes.append(p)

    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, shared_model, counter, lock, mp_rewards, optimizer, not have_wts, env_arguments, finish_signal))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

