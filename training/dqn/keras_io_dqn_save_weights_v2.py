## This entire file has been adapted from code
## by  Jacob Chapman and Mathias Lechner, available here as of 
## 11/11/2021
## https://github.com/keras-team/keras-io/blob/master/examples/rl/deep_q_network_breakout.py
## Changes made by Matthew Yee-King:
## * use breakwall version of breakout instead of atari
## * log in tensorboard compatible format and print logs
## * save weights of model each time moving episodic reward reaches a new max
## * save_weights function adjusted to work better on windows
## Changes made by Alexander Polyakov
## * load_weights function added to load from a checkpoint
## * model training moved to a separate function
## * replay history is fixed to be memory efficient allowing 1kk frames taking slightly less than 32GB of RAM
## * memory leak fixed in model_target, as when predict() is used it allocates new scene graph each frame
## * ring buffer is added for replay history to improve its efficiency
## * build_name argument is added to specify where to save to and load from model weights

from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.atari_wrappers import MaxAndSkipEnv
import numpy as np
import tensorflow as tf
import gymnasium
from tensorflow import keras
from tensorflow.keras import layers
import datetime
import os 
import json
import gc, psutil
import objgraph
import tracemalloc
import argparse
import time

import sys

sys.path.append("../../game")

parser = argparse.ArgumentParser(description='DQN')
parser.add_argument('--build_name')
args = parser.parse_args()


# logging code
# for tensorboard
# https://www.tensorflow.org/tensorboard/get_started
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
# train_summary_writer = tf.summary.create_file_writer(train_log_dir)
# end of  logging code
# for tensorboard

# Configuration paramaters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 10000

#env_name = "BreakoutNoFrameskip-v4"
env_name = "breakwall_clone:breakwall"

# Use the Baseline Atari environment because of Deepmind helper functions
env = MaxAndSkipEnv(gymnasium.make(env_name), skip=4)

# Warp the frames, grey scale, stake four frame and scale to smaller ratio
env = wrap_deepmind(env, frame_stack=True, scale=False)
env.seed(seed)

"""
## Implement the Deep Q-Network

This network learns an approximation of the Q-table, which is a mapping between
the states ainnd actions that an agent will take. For every state we'll have four
actions, that can be taken. The environment provides the state, and the action
is chosen by selecting the larger of the four Q-values predicted in the output layer.
"""

num_actions = 4


def log(running_reward, last_reward, episode, mem_perc, epsilon, frame, tensorboard_log = False):
    """
    log the running episodic reward, most recent reward, 
    episode count, epsilon value and frame count plus mem_perc which
    is the percentage of the action memory that is full
    """
    if tensorboard_log: 
        with train_summary_writer.as_default():
            tf.summary.scalar('running reward', running_reward, step=episode)

    template = 'Epoch,{}, Mem,{}%, Eps,{}, Frame,{}, Last reward:,{}, Running reward:,{}, '
    print (template.format(episode+1,
                    np.round(mem_perc, 3), 
                    np.round(epsilon, 3), 
                    frame,
                    last_reward, 
                    running_reward))

def load_weights(build, model, target):
    print("trying to load weights and history")
    path = os.path.join('saves', build + "_model.ckpt")
    target_path = os.path.join('saves', build + "_target.ckpt")
    if os.path.exists(path + '.index') and os.path.exists(target_path + '.index'):
        model.load_weights(path)
        target.load_weights(target_path)

def save_weights(build, model, target):
    """
    save the weights of the sent model 
    with the env_name, episode and run_id used 
    to gneerate the filename
    """
    print("Saving weights")
    path = os.path.join('saves', build + "_model.ckpt")
    target_path = os.path.join('saves', build + "_target.ckpt")
    model.save_weights(path)
    target.save_weights(target_path)

def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(84, 84, 4,))
    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)
    layer4 = layers.Flatten()(layer3)
    layer5 = layers.Dense(512, activation="relu")(layer4)    
    action = layers.Dense(num_actions, activation="linear")(layer5)
    return keras.Model(inputs=inputs, outputs=action)

denom = 1.0 / 255.0
def train():
    indices = np.random.choice(range(len(replay_history)), size=batch_size)

    # Using list comprehension to sample from replay buffer
    
    actions, states, states_next, rewards, dones = [], [], [], [], []
    for i in indices:
        act, state, st_next, reward, done = replay_history[i]
        actions.append(act)
        states.append(state)
        states_next.append(st_next)
        rewards.append(reward)
        dones.append(done)
    action_sample = tf.convert_to_tensor(np.array(actions))
    state_sample = tf.convert_to_tensor(np.array(states) * denom, dtype=np.float32)
    state_next_sample = tf.convert_to_tensor(np.array(states_next) * denom, dtype=np.float32)
    rewards_sample = tf.convert_to_tensor(np.array(rewards), dtype=np.float32)
    done_sample = tf.convert_to_tensor(np.array(dones), dtype=np.float32)

    # Build the updated Q-values for the sampled future states
    # Use the target model for stability
    future_rewards = model_target(state_next_sample, training=False)
    # Q value = reward + discount factor * expected future reward
    updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1)

    # If final frame set the last value to -1
    updated_q_values = updated_q_values * (1 - done_sample) - done_sample

    # Create a mask so we only calculate loss on the updated Q-values
    masks = tf.one_hot(action_sample, num_actions)

    with tf.GradientTape() as tape:
        # Train the model on the states and updated Q-values
        q_values = model(state_sample)

        # Apply the masks to the Q-values to get the Q-value for action taken
        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
        # Calculate loss between new Q-value and old Q-value
        loss = loss_function(updated_q_values, q_action)

    # Backpropagation
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

# The first model makes the predictions for Q-values which are used to
# make a action.
model = create_q_model()
print(model.summary())
# Build a target model for the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.
model_target = create_q_model()

"""
## Train
"""
# In the Deepmind paper they use RMSProp however then Adam optimizer
# improves training time
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

# Experience replay buffers
replay_history = []
load_weights(args.build_name, model, model_target)
repl_idx = 0

episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0
# Number of frames to take random action and observe output
epsilon_random_frames = 50000
# Number of frames for exploration
epsilon_greedy_frames = 1000000.0
# Maximum replay length
max_memory_length = 1000000
# Train the model after 4 actions
update_after_actions = 4
# How often to update the target network
update_target_network = 10000
# Using huber loss for stability
loss_function = keras.losses.Huber()
epoch = 0
# use this to decide when it is time to save the weights
max_reward = 0 

def save_metrics(metrics, build):
    path = os.path.join("metrics", build + ".json")
    with open(path, 'w') as f:
        json.dump(metrics, f)

metrics = []

timer = 0.0
while True:  # Run until solved
    state, _ = env.reset()
    episode_reward = 0
    print("Frame", frame_count, "Episode", episode_count)
    for timestep in range(1, max_steps_per_episode): #10000
        #env.render()# ; Adding this line would show the attempts
        # of the agent in a pop up window.
        if frame_count % 250 == 0:
            metric = {
                "time": time.process_time(),
                "mean_reward": np.mean(episode_reward_history),
                "frames": frame_count
            }
            metrics.append(metric)
            log(np.mean(episode_reward_history), 
                episode_reward, 
                episode_count, 
                len(replay_history) / max_memory_length * 100, 
                epsilon, 
                frame_count)
            save_metrics(metrics, args.build_name)

        frame_count += 1
        # Use epsilon-greedy for exploration
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            # Take random action
            action = np.random.choice(num_actions)
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(np.array(state) * denom, dtype = np.float32)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()

        # Decay probability of taking random action
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        # Apply the sampled action in our environment
        state_next, reward, done, trunc, _ = env.step(action)

        #print("state shape:", state.shape)
        episode_reward += reward

        # Save actions and states in replay buffer
        # Limit the state and reward history
        if len(replay_history) == max_memory_length:
            replay_history[repl_idx] = (action, state, state_next, reward, done)
            repl_idx = (repl_idx + 1) % max_memory_length
        else:
            replay_history.append( (action, state, state_next, reward, done) )
        state = state_next

        # Update every fourth frame and once batch size is over 32
        if frame_count % update_after_actions == 0 and len(replay_history) > batch_size:
            train()

        if frame_count % update_target_network == 0:
            print("upd weights")
            # update the the target network with new weights
            model_target.set_weights(model.get_weights())


        if done:
            break

    # Update running reward to check condition for solving
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    episode_count += 1

    # save the weights if we've reached a new high
    if running_reward > max_reward:
        save_weights(args.build_name, model, model_target)
        max_reward = running_reward

    if running_reward > 75:  # Condition to consider the task solved
        save_weights(args.build_name, model, model_target)
        print("Solved at episode {}!".format(episode_count))
        break
