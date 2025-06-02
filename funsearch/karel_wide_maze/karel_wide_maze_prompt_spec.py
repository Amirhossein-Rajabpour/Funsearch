"""Specification for the Karel Wide Maze environment.

We are searching for a function `get_action(obs)` that returns an action (int) for the Karel environment.

The observation is a vector of 4 floats (0.0 or 1.0) representing:
  [frontIsClear, leftIsClear, rightIsClear, markersPresent]

The actions are:
  0: move
  1: turnLeft
  2: turnRight
  3: pickMarker (not used in maze)
  4: putMarker (not used in maze)

In the maze task, the agent starts at a fixed position and must navigate to a goal marker. The environment uses a sparse reward: 1 when reaching the goal, 0 otherwise.

The environment is a wide maze (corridors are 2 cells wide) and has multiple initial configurations (both mazes and goal positions).
"""

import numpy as np
import random
# from karel_gym import KarelGymEnv
from gym_envs.karel_gym import KarelGymEnv  

# Configuration for the environment
ENV_CONFIG = {
    'task_name': 'maze',
    'env_height': 20,
    'env_width': 20,
    'max_steps': 1000,
    'sparse_reward': True,
    'seed': 42,
    'multi_initial_confs': True, 
    'all_initial_confs': True,
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
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    return total_reward

@funsearch.run
def evaluate(n: int) -> float:
    """Returns the success rate over n episodes."""
    successes = 0
    for i in range(n):
        env = make_env(seed=i)
        total_reward = run_episode(env, get_action, seed=i)
        if total_reward > 0:
            successes += 1
    return successes / n

@funsearch.evolve
def get_action(obs: np.ndarray) -> int:
    """Returns the action to take given the observation."""
    # Random policy (to be evolved by FunSearch)
    return random.randint(0, 4)