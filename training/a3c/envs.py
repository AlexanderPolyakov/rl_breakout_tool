# Based on
# https://github.com/ikostrikov/pytorch-a3c
import gymnasium
from gymnasium import spaces
import numpy as np
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

from baselines.common.atari_wrappers import MaxAndSkipEnv

# Taken from https://github.com/openai/universe-starter-agent
def create_atari_env(env_id, env_arguments):
    env = gymnasium.make(env_id, **env_arguments)
    env = MaxAndSkipEnv(env, skip=4)
    env = wrap_deepmind(env, frame_stack=True, scale=True)
    env = ArrayEnv(env)
    return env

class ArrayEnv(gymnasium.ObservationWrapper):
    def __init__(self, env=None):
        super(ArrayEnv, self).__init__(env)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=1, shape=(shp[-1], shp[0], shp[1]), dtype=env.observation_space.dtype)

    def observation(self, observation):
        return np.transpose(np.array(observation), (2, 0, 1))

