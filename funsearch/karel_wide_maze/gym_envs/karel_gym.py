import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import gymnasium as gym
import numpy as np
from typing import Optional, Callable
import random

# from environment.karel_env.karel_tasks.maze import Maze, MazeSparse, MazeSparseAllInit, MazeAllInit, MazeWide, MazeWideSparseAllInit, MazeWideSparse
from karel_wide_maze.karel_tasks.maze import Maze, MazeSparse, MazeSparseAllInit, MazeAllInit, MazeWide, MazeWideSparseAllInit, MazeWideSparse


class KarelGymEnv(gym.Env):
    """
    Gym environment wrapper for the KarelEnvironment.
    """
    metadata = {'render.modes': ['human', 'ansi']}
    SUPPORTED_TASKS = ['maze', 'maze_sparse']

    def __init__(self, env_config: Optional[dict] = None, options: Optional[list] = None):
        super(KarelGymEnv, self).__init__()

        default_config = {
            'task_name': 'maze',
            'env_height': 20,
            'env_width': 20,
            'max_steps': 1000,
            'sparse_reward': True,
            'seed': None,
            'initial_state': None
        }

        if env_config is not None:
            default_config.update(env_config)
        self.config = default_config

        self._handle_initial_state()

        # Set random seed
        self.seed(self.config['seed'])

        if self.config['task_name'] not in self.SUPPORTED_TASKS:
            raise ValueError(f"Task {self.config['task_name']} not supported. "
                           f"Choose from {self.SUPPORTED_TASKS}")

        # Initialize environment variables
        self.env_height = self.config['env_height']
        self.env_width = self.config['env_width']
        self.max_steps = self.config['max_steps']
        self.current_step = 0
        self.reward_diff = self.config['reward_diff'] if 'reward_diff' in self.config else False
        self.rescale_reward = self.config['reward_scale'] if 'reward_scale' in self.config else True
        self.multi_initial_confs = self.config['multi_initial_confs'] if 'multi_initial_confs' in self.config else False
        self.all_initial_confs = self.config['all_initial_confs'] if 'all_initial_confs' in self.config else False
        self.all_initial_confs_envs = None
        self.last_action = -1.0

        # Initialize the task
        self.task_name = self.config['task_name']
        self.task, self.task_specific = self._initialize_task()

        self._set_action_observation_spaces(options)
        self.reset()

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)

    def _initialize_task(self):
        env_args = {
            'env_height': self.env_height,
            'env_width': self.env_width,
            'crashable': False,
            'leaps_behaviour': False,
        }

        if self.task_name == 'maze':
            if self.all_initial_confs and not self.config.get('wide_maze', False):
                task_class = MazeSparseAllInit if self.config['sparse_reward'] else MazeAllInit
                task_specific = task_class(env_args=env_args)
                self.all_initial_confs_envs = task_specific.all_initial_confs
            elif self.all_initial_confs and self.config.get('wide_maze', False) and self.config['sparse_reward']:
                task_specific = MazeWideSparseAllInit(
                    env_args=env_args,
                    seed=self.config.get('seed'),
                )
                self.all_initial_confs_envs = task_specific._initial_confs
            elif self.config.get('wide_maze', False):
                task_class = MazeWideSparse if self.config['sparse_reward'] else MazeWide
                task_specific = task_class(
                    env_args=env_args,
                    seed=self.config.get('seed'),
                )
            else:
                task_class = MazeSparse if self.config['sparse_reward'] else Maze
                task_specific = task_class(
                    env_args=env_args,
                    seed=self.config.get('seed'),
                )
            task = task_specific.generate_initial_environment(env_args)

        else:
            raise ValueError(f"Unknown task: {self.task_name}")

        return task, task_specific

    def _set_action_observation_spaces(self, options: Optional[list] = None):
        # num_features = self.task.state_shape[0]
        observation_shape = self._get_observation_dsl().shape        
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=observation_shape,
            dtype=np.float32
        )

        if options is not None:
            pass
        else:
            self.action_space = gym.spaces.Discrete(len(self.task.actions_list))
            self.program_stack = None
            self.option_index = None


    def step(self, action:int):
        assert self.action_space.contains(action), "Invalid action"
        self.last_action = action
        truncated = False
        def process_action(action:int):
            nonlocal truncated
            action_name = self.task.actions_list[action]
            self.task.run_action(action_name)

            self.current_step += 1

            if self.task_name != 'base':
                terminated, reward = self.task_specific.get_reward(self.task)

            if self.current_step >= self.max_steps:
                truncated = True
            

            return self._get_observation_dsl(), reward, terminated, truncated, {}
        return process_action(action)

    def reset(self, seed=0, options=None):
        self.current_step = 0
        self.last_action = -1.0
        
        if self.multi_initial_confs:   # choose between 10 random initial setups
            selected_seed = random.choice(list(range(10)))
            self.config['seed'] = selected_seed
            self.seed(selected_seed)
            self.task, self.task_specific = self._initialize_task()

        elif self.all_initial_confs:
            selected_conf = random.choice(self.all_initial_confs_envs)
            self.config['initial_state'] = selected_conf.copy()
            env_args = {
                'env_height': self.env_height,
                'env_width': self.env_width,
                'crashable': False,
                'leaps_behaviour': False,
                'initial_state': self.config.get('initial_state')
            }
            self.task = self.task_specific.generate_initial_environment(env_args)

        else:
            self.task, self.task_specific = self._initialize_task()

        # self.task.state2image(root_dir=project_root + '/environment/').show()
        return self._get_observation_dsl(), {}

    def render(self, mode='human'):
        if mode == 'human':
            print(self.task.to_string(), "\n")
        elif mode == 'ansi':
            return self.task.to_string()
        else:
            super(KarelGymEnv, self).render(mode=mode)

    def get_observation(self) -> np.ndarray:
        return self.task.get_state().astype(np.float32)

    def _get_observation_dsl(self) -> np.ndarray:
        """
        Returns an observation that a DSL agent would see but for our RL agent
        """
        num_actions = 5 + 1 # number of actions + 1
        one_hot_action = np.zeros(num_actions, dtype=float)
        
        if self.last_action is not None and self.last_action != -1: 
            one_hot_action[int(self.last_action) + 1] = 1.0
        elif self.last_action == -1:
            one_hot_action[0] = 1.0

        dsl_obs = np.array([
            self.task.get_bool_feature("frontIsClear"),
            self.task.get_bool_feature("leftIsClear"),
            self.task.get_bool_feature("rightIsClear"),
            self.task.get_bool_feature("markersPresent"),
        ], dtype=float)

        # dsl_obs = np.concatenate((dsl_obs, one_hot_action))

        return dsl_obs

    def _handle_initial_state(self):
        initial_state = self.config.get('initial_state')
        if initial_state is not None:
            # Extract dimensions from initial_state
            if isinstance(initial_state, np.ndarray):
                num_features, env_height, env_width = initial_state.shape
            else:
                raise ValueError("initial_state must be a NumPy array")

            self.env_height = env_height
            self.env_width = env_width

            self.config['env_height'] = env_height
            self.config['env_width'] = env_width
        else:
            self.env_height = self.config['env_height']
            self.env_width = self.config['env_width']


def make_karel_env(env_config: Optional[dict] = None) -> Callable:
    """
    Factory function to create a KarelGymEnv instance with the given configuration.
    """
    def thunk():
        env = KarelGymEnv(env_config=env_config)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


if __name__ == "__main__":

    num_features = 16
    env_height = 12
    env_width = 12

    # A custom initial state for the base task
    initial_state = np.zeros((num_features, env_height, env_width), dtype=bool)
    initial_state[1, 0, 0] = True  # Karel facing East at (0, 0)
    initial_state[4, 1, 2] = True  # Wall at (1, 2)

    env_config = {
        'task_name': 'maze',
        'env_height': env_height,
        'env_width': env_width,
        'max_steps': 1,
        'sparse_reward': True,
        'seed': 50,
        'initial_state': None,
        'multi_initial_confs': False, 
        'all_initial_confs': True,
        'wide_maze': True,
    }

    env = make_karel_env(env_config=env_config)()

    # # showing all the initial configurations
    # print("len of all initial confs:", len(env.all_initial_confs_envs))
    # for init_conf in env.all_initial_confs_envs:
    #     env.task.state2image(init_conf, root_dir=project_root + '/environment/').show()
    # exit()

    init_obs = env.reset()
    env.render()
    # env.task.state2image(env.get_observation(), root_dir=project_root + '/environment/').show()

    action_names = env.task.actions_list
    action_mapping = {name: idx for idx, name in enumerate(action_names)}
    action_sequence = ['move', 'turnLeft', 'move', 'move', 'turnRight', 'move', 'turnLeft', 'move', 'turnRight', 'move'] # for stairclimber 6*6
    
    actions = [action_mapping[name] for name in action_sequence]

    done = False
    total_reward = 0
    for action in actions:
        print("--- Action:", action_names[action])
        obs, reward, done, truncated, info = env.step(action)
        print("--- Reward:", reward)
        print("--- Done:", done)
        total_reward += reward
        env.render()
        env.task.state2image(env.get_observation(), root_dir=project_root + '/environment/').show()
        if done or truncated:
            print("Episode done")
            break
    print("Total Reward:", total_reward)

    reset_obs = env.reset()
    env.render()
