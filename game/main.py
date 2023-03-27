import gymnasium as gym
from gymnasium.utils.play import play

env = gym.make('breakwall_clone:breakwall', render_mode="rgb_array")
play(env, keys_to_action={ "a": 3, "d": 2 }, noop=0, zoom=2)
