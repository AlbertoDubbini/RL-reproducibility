from typing import List

import gym
import numpy as np
from tqdm import tqdm

from ..agent.agent import Agent

import time

def run(env: gym.Env, agent: Agent, num_episodes: int, n_rollouts: int) -> List[float]:
    cum_rewards = []
    for i_episode in tqdm(range(num_episodes)):
        cum_reward = 0
        state = env.reset()
        idx = 0
        while True:
            # print("env step: ", idx)
            action, cum_action_vals = agent.act(env, state, n_rollouts)
            state, reward, done, info = env.step(action)
            if "BlackJack" in str(env):
                cum_reward += np.mean(cum_action_vals)
            else:
                cum_reward += reward 
            idx += 1
            # print("action: ",action)                
            # print("state: ",state)         
            # print("cum_reward: ",cum_reward)
            if done:
                break
        # print("end\n")
        cum_rewards.append(cum_reward)
    env.close()
    return cum_rewards