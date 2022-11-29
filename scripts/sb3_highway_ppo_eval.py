import gym
import torch as th
from stable_baselines3 import PPO
from torch.distributions import Categorical
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
import highway_env

model = PPO.load("highway_ppo/model")
env = gym.make("highway-fast-v0")
for _ in range(5):
    #obs, info = env.reset()
    obs = env.reset()
    done = truncated = False
    while not (done or truncated):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(int(action))
        env.render()