# Based on
# https://github.com/ikostrikov/pytorch-a3c
import time
from collections import deque

import torch
import torch.nn.functional as F

from envs import create_atari_env
from model import ActorCritic

import os
import json

import numpy as np

import cv2 as cv

def save_metrics(metrics, build):
    path = os.path.join("metrics", build + ".json")
    with open(path, 'w') as f:
        json.dump(metrics, f)

def save_model_weights(state_dict, build):
    torch.save(state_dict, "checkpoints/" + build + ".pt")

def test(rank, args, shared_model, counter, lock, mp_rewards, init_wts, env_arguments, finish_signal):
    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name, env_arguments)
    env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space, init_wts)

    model.eval()

    state, _ = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=10000)
    episode_length = 0
    best_rew = 0
    frames = []
    metrics = []
    best_rewards = None
    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())

        with torch.no_grad():
            value, logit = model(state.unsqueeze(0))
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1, keepdim=True)[1].numpy()

        state, reward, done, tr, _ = env.step(action[0, 0])

        if args.eval_mode:
            frames.append(env.game.getScreenRGB())

        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            with lock:
                if len(mp_rewards) > 0:
                    mean_rewards = np.mean(mp_rewards)
                else:
                    mean_rewards = 0.0
            metric = {
                "time": time.time() - start_time,
                "mean_reward": mean_rewards,
                "frames": counter.value
            }
            metrics.append(metric)
            save_metrics(metrics, args.build_name)
            if best_rewards is not None:
                if mean_rewards > best_rewards:
                    if not args.eval_mode:
                        save_model_weights(shared_model.state_dict(), args.build_name)
                    best_rewards = mean_rewards
            else:
                best_rewards = mean_rewards

            print("Time {}, num steps {}, FPS {:.0f}, mean episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                counter.value, counter.value / (time.time() - start_time),
                mean_rewards, episode_length))

            if reward_sum > best_rew and args.eval_mode:
                best_rew = reward_sum

                fourcc = cv.VideoWriter_fourcc(*'mp4v')
                out_video = cv.VideoWriter('output/' + args.build_name + '.mp4', fourcc,
                                     60.0,
                                     (env.game.screen_width,  env.game.screen_height),
                                     True)

                for frame in frames:
                    out_video.write(frame)

                eval_len = len(frames) // 60
                if eval_len >= args.eval_len:
                    prev_res = []
                    with open("times.json", "r") as initf:
                        prev_res = json.load(initf)
                    prev_res.append(time.time() - start_time)
                    with open("times.json", "w") as resf:
                        json.dump(prev_res, resf)
                    finish_signal.value = 1
                    break
                out_video.release()
            
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state, _ = env.reset()

            time.sleep(30 if args.eval_mode else 60)

            frames = []
            
        state = torch.from_numpy(state)

