"""Plays the Karel Wide Maze environment."""
import numpy as np
import random
# from karel_gym import KarelGymEnv  # Import from your existing implementation
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
        if total_reward > 0:  # Sparse reward: 1 for success, 0 otherwise
            successes += 1
    return successes / n

@funsearch.evolve
def get_action(obs: np.ndarray) -> int:
    """Returns the action to take given the observation."""
    # Random policy (to be evolved)
    return random.randint(0, 4)  # 5 possible actions