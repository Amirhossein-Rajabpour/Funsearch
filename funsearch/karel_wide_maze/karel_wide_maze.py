"""Plays the Karel Wide Maze environment."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))

import numpy as np
import random
from typing import Union
from karel_wide_maze.gym_envs.karel_gym import KarelGymEnv

from util import LOG

# Configuration for the environment
ENV_CONFIG = {
    'task_name': 'maze',
    'env_height': 20,
    'env_width': 20,
    'max_steps': 1000,
    'sparse_reward': True,
    'seed': None,
    'multi_initial_confs': False, 
    'all_initial_confs': False,
    'wide_maze': True,
}

def make_env(seed=None):
  """Create a Karel Wide Maze environment."""
  config = ENV_CONFIG.copy()
  config['seed'] = seed
  return KarelGymEnv(env_config=config)

def run_episode(env, get_action, seed=None):
  """Run one episode and return the total reward."""
  obs, info = env.reset(seed=seed)
  total_reward = 0
  actions = get_action(env)
  for action in actions:
      obs, reward, terminated, truncated, info = env.step(action)
      total_reward += reward
  return total_reward

@funsearch.run
def evaluate(n: int) -> float:
  """Evaluates the performance of the Karel agent in the environment on 100 random initial configurations."""
  successes = 0
  for i in range(n):
      env = make_env(seed=i)
      total_reward = run_episode(env, get_action, seed=i)
      if total_reward > 0:  # Sparse reward: 1 for success, 0 otherwise
          successes += 1
  return successes / n

@funsearch.evolve
def get_action(env: KarelGymEnv) -> Union[list[int], str]:
  """Creates a policy that returns a list of actions for the Karel agent to take in the environment."""
  return [random.randint(0, 4) for _ in range(50)] 