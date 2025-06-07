"""Specification for the Karel Wide Maze environment.

We are searching for a function `get_action(env)` that returns a list of actions list[int] for the Karel environment.
get_action(env) should return a policy that can solve the maze all the time, regardless of the initial configuration. Then by calling this policy, it can get the actions for that specific initial configuration.

Input is a KarelGymEnv object.
- You can access the height and width of the env like this: env.env_width, env.env_height
- You can access walls like this:
  # static walls from feature index 4 of the Karel state
  state_arr = env.task.get_state()             # shape: (features, H, W)
  self.walls = state_arr[4].astype(bool)       # True where wall
- You can the row, column, and direction of the agenr like this:
  r, c, d = env.task.get_hero_pos()
- And the directions are like this:
  0: 'Karel facing North',
  1: 'Karel facing East',
  2: 'Karel facing South',
  3: 'Karel facing West',
- Access the goal marker position like this:
  goal_r, goal_c = env.task_specific.marker_position
- You can access the observation like this:
  obs = env._get_observation_dsl()  # shape: (4,), [frontIsClear, leftIsClear, rightIsClear, markersPresent]


The actions are:
  0: move
  1: turnLeft
  2: turnRight
  3: pickMarker (not used in maze)
  4: putMarker (not used in maze)

In the maze task, the agent starts at a fixed position and must find its path to a goal marker. The environment uses a sparse reward: 1 when reaching the goal, 0 otherwise.
The environment is a wide maze (corridors are 2 cells wide) and has multiple initial configurations (both mazes and goal positions).



This specification describes the key classes, variables, and functions used to
define a Gym‐compatible “Karel Wide Maze” task, where an agent navigates a 
carved maze to reach a goal marker under sparse rewards.



Package Layout:
  project_root/
  └── funsearch/
      ├── implementation/
      │   ├── utils.py     
      │   └── temp1.py      # top‐level script that calls evaluate and get_action
      ├── karel_wide_maze/
      │   ├── __init__.py
      │   ├── karel_wide_maze.py
      │   ├── karel_wide_maze_prompt_spec.py
      │   ├── gym_envs/
      │   │   ├── __init__.py
      │   │   └── karel_gym.py     # Defines KarelGymEnv
      │   ├── karel_tasks/
      │   │   ├── __init__.py
      │   │   └── maze.py          # Defines Maze, MazeSparse, MazeWide, etc.
      │   ├── karel/
      │   │   ├── __init__.py
      │   │   └── environment.py   # Defines KarelEnvironment and features
      │   └── base/
      │       ├── __init__.py
      │       └── task.py          # Defines BaseTask
      └──...

Usage Summary:
  1. The FunSearch framework “evolves” a Python function `get_action(env)` to maximize `evaluate(n)`.
  2. `evaluate(n)` runs n episodes of the Karel Wide Maze environment, each seeded differently, with different locations for walls and goal.
  3. Each episode calls `run_episode()`, which repeatedly:
     - Queries `get_action(env)` to obtain actions ∈ {0..4}.
     - Steps the Gym environment and accumulates sparse/dense rewards.
     - Terminates when Karel reaches the goal or max_steps is reached.
  4. Maze‐classes in karel_tasks/maze.py carve out a random maze layout (via DFS), set a goal marker,
     and compute rewards either sparsely (1 upon reach) or densely (normalized distance progress).
  5. KarelGymEnv wraps these Tasks into a standard Gym API: it exposes `step()`, `reset()`, `render()`,
     `action_space`, `observation_space`, and handles “multiple initial configurations” if requested.

"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))

import numpy as np
import random
from typing import Union

from karel_wide_maze.gym_envs.karel_gym import KarelGymEnv  

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