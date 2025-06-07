import numpy as np

from ..base import BaseTask
from ..karel import KarelEnvironment


import os
import importlib.util


utils_path = os.path.abspath(
    os.path.join(__file__, "..", "..", "..", "implementation", "util.py")
)
spec = importlib.util.spec_from_file_location("util", utils_path)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)
LOG = utils.LOG



class Maze(BaseTask):
        
    def generate_initial_environment(self, env_args):
        
        reference_env = KarelEnvironment(**env_args)
        
        env_height = reference_env.state_shape[1]
        env_width = reference_env.state_shape[2]        
        
        def get_neighbors(cur_pos):
            neighbor_list = []
            #neighbor top
            if cur_pos[0] - 2 > 0: neighbor_list.append([cur_pos[0] - 2, cur_pos[1]])
            # neighbor bottom
            if cur_pos[0] + 2 < env_height - 1: neighbor_list.append([cur_pos[0] + 2, cur_pos[1]])
            # neighbor left
            if cur_pos[1] - 2 > 0: neighbor_list.append([cur_pos[0], cur_pos[1] - 2])
            # neighbor right
            if cur_pos[1] + 2 < env_width - 1: neighbor_list.append([cur_pos[0], cur_pos[1] + 2])
            return neighbor_list
        
        state = np.zeros(reference_env.state_shape, dtype=bool)
        state[4, :, :] = True
        
        init_pos = [env_height - 2, 1]
        state[1, init_pos[0], init_pos[1]] = True
        state[4, init_pos[0], init_pos[1]] = False
        visited = np.zeros((env_height, env_width), dtype=bool)
        visited[init_pos[0], init_pos[1]] = True
        
        stack = [init_pos]
        while len(stack) > 0:
            cur_pos = stack.pop()
            neighbors = get_neighbors(cur_pos)
            self.rng.shuffle(neighbors)
            for neighbor in neighbors:
                if not visited[neighbor[0], neighbor[1]]:
                    visited[neighbor[0], neighbor[1]] = True
                    state[4, (cur_pos[0] + neighbor[0]) // 2, (cur_pos[1] + neighbor[1]) // 2] = False
                    state[4, neighbor[0], neighbor[1]] = False
                    stack.append(neighbor)
        
        valid_loc = False
        state[5, :, :] = True
        while not valid_loc:
            ym = self.rng.randint(1, env_height - 1)
            xm = self.rng.randint(1, env_width - 1)
            if not state[4, ym, xm] and not state[1, ym, xm]:
                valid_loc = True
                state[6, ym, xm] = True
                state[5, ym, xm] = False
                self.marker_position = [ym, xm]
        
        self.initial_distance = abs(init_pos[0] - self.marker_position[0]) \
            + abs(init_pos[1] - self.marker_position[1])
        
        return KarelEnvironment(initial_state=state, **env_args)
    
    def reset_environment(self):
        super().reset_environment()
        self.previous_distance = self.initial_distance

    def get_reward(self, env: KarelEnvironment):
        terminated = False
        reward = 0.

        karel_pos = env.get_hero_pos()
        
        current_distance = abs(karel_pos[0] - self.marker_position[0]) \
            + abs(karel_pos[1] - self.marker_position[1])
        
        # Reward is how much closer Karel is to the marker, normalized by the initial distance
        reward = (self.previous_distance - current_distance) / self.initial_distance

        if karel_pos[0] == self.marker_position[0] and karel_pos[1] == self.marker_position[1]:
            terminated = True
        
        self.previous_distance = current_distance
        
        return terminated, reward

class MazeSparse(Maze):
    
    def get_reward(self, env: KarelEnvironment):
        terminated = False
        reward = 0.

        karel_pos = env.get_hero_pos()
        
        if karel_pos[0] == self.marker_position[0] and karel_pos[1] == self.marker_position[1]:
            terminated = True
            reward = 1.
            print("** Agent reached the goal!!!!")
        
        return terminated, reward
    

class MazeWide(Maze):
            
    def generate_initial_environment(self, env_args):

        path_width = 2
        # compute how far to carve on each side of the centerline
        # e.g. for width=1 => offsets=[0]
        #      for width=2 => half=1=> offsets=[-1,0]
        #      for width=3 => half=1=> offsets=[-1,0,1]
        floor_half = path_width // 2
        ceil_half = path_width - floor_half - 1
        offsets = range(-floor_half, ceil_half + 1)
        
        reference_env = KarelEnvironment(**env_args)
        
        env_height = reference_env.state_shape[1]
        env_width = reference_env.state_shape[2]        
        
        def get_neighbors(cur_pos):
            neighbor_list = []
            #neighbor top
            if cur_pos[0] - 2 > 0: neighbor_list.append([cur_pos[0] - 2, cur_pos[1]])
            # neighbor bottom
            if cur_pos[0] + 2 < env_height - 1: neighbor_list.append([cur_pos[0] + 2, cur_pos[1]])
            # neighbor left
            if cur_pos[1] - 2 > 0: neighbor_list.append([cur_pos[0], cur_pos[1] - 2])
            # neighbor right
            if cur_pos[1] + 2 < env_width - 1: neighbor_list.append([cur_pos[0], cur_pos[1] + 2])
            return neighbor_list
        
        state = np.zeros(reference_env.state_shape, dtype=bool)
        state[4, :, :] = True
        init_pos = [env_height - 2, 1]
        init_r, init_c = init_pos[0], init_pos[1]
        state[1, init_r, init_c] = True
        for dr in offsets:
            for dc in offsets:
                rr, cc = init_r + dr, init_c + dc
                if 0 <= rr < env_height and 0 <= cc < env_width:
                    state[4, rr, cc] = False

        # set up DFS
        visited = np.zeros((env_height, env_width), dtype=bool)
        visited[init_r, init_c] = True
        stack = [[init_r, init_c]]

        while stack:
            cur_r, cur_c = stack.pop()
            nbrs = get_neighbors([cur_r, cur_c])
            self.rng.shuffle(nbrs)
            for nbr_r, nbr_c in nbrs:
                if visited[nbr_r, nbr_c]:
                    continue
                visited[nbr_r, nbr_c] = True

                # carve a thick corridor between (cur) and (nbr)
                if cur_r == nbr_r:  # horizontal move
                    c0, c1 = sorted([cur_c, nbr_c])
                    for dr in offsets:
                        rr = cur_r + dr
                        if not (0 <= rr < env_height): 
                            continue
                        for cc in range(c0, c1 + 1):
                            state[4, rr, cc] = False
                else:  # vertical move
                    r0, r1 = sorted([cur_r, nbr_r])
                    for dc in offsets:
                        cc = cur_c + dc
                        if not (0 <= cc < env_width):
                            continue
                        for rr in range(r0, r1 + 1):
                            state[4, rr, cc] = False

                stack.append([nbr_r, nbr_c])

        # re-enforce the outer border as walls
        state[4, 0, :] = True
        state[4, env_height - 1, :] = True
        state[4, :, 0] = True
        state[4, :, env_width - 1] = True
        
        valid_loc = False
        state[5, :, :] = True
        while not valid_loc:
            ym = self.rng.randint(1, env_height - 1)
            xm = self.rng.randint(1, env_width - 1)
            if not state[4, ym, xm] and not state[1, ym, xm]:
                valid_loc = True
                state[6, ym, xm] = True
                state[5, ym, xm] = False
                self.marker_position = [ym, xm]
        
        self.initial_distance = abs(init_pos[0] - self.marker_position[0]) \
            + abs(init_pos[1] - self.marker_position[1])
        
        return KarelEnvironment(initial_state=state, **env_args)
    
    def reset_environment(self):
        super().reset_environment()
        self.previous_distance = self.initial_distance

    def get_reward(self, env: KarelEnvironment):
        terminated = False
        reward = 0.

        karel_pos = env.get_hero_pos()
        
        current_distance = abs(karel_pos[0] - self.marker_position[0]) \
            + abs(karel_pos[1] - self.marker_position[1])
        
        # Reward is how much closer Karel is to the marker, normalized by the initial distance
        reward = (self.previous_distance - current_distance) / self.initial_distance

        if karel_pos[0] == self.marker_position[0] and karel_pos[1] == self.marker_position[1]:
            terminated = True
        
        self.previous_distance = current_distance
        
        return terminated, reward

class MazeWideSparse(MazeWide):

    def get_reward(self, env: KarelEnvironment):
        terminated = False
        reward = 0.

        karel_pos = env.get_hero_pos()
        
        if karel_pos[0] == self.marker_position[0] and karel_pos[1] == self.marker_position[1]:
            terminated = True
            reward = 1.0
            # print("** Agent reached the goal!!!!")
        
        return terminated, reward
    

class MazeWideSparseAllInit(BaseTask):
    """
    Like MazeWide, but generates `num_mazes` different wide mazes
    and presents each (maze + marker) configuration in turn with a sparse reward.
    """
    def __init__(self, env_args, seed=None, num_mazes=5):
        # initialize BaseTask (via MazeWide) to set up rng and env_args
        print(f"** MazeWideSparse: generating {num_mazes} mazes")
        super().__init__(env_args, seed=seed)
        self.done = False
        self.previous_distance = None
        self.num_mazes = num_mazes
        self.seed = 0 if seed is None else seed
        
        # pre-generate all wide-maze + marker configs
        self._initial_confs = self._generate_all_confs(env_args)

    def _generate_all_confs(self, env_args):
        """
        Carve `num_mazes` unique wide mazes and for each one,
        generate a separate state array for every possible marker location.
        Returns a list of boolean state arrays.
        """
        reference_env = KarelEnvironment(**env_args)
        H, W = reference_env.state_shape[1], reference_env.state_shape[2]

        # wide-maze carving parameters from MazeWide
        path_width = 2
        floor_half = path_width // 2
        ceil_half = path_width - floor_half - 1
        offsets = range(-floor_half, ceil_half + 1)

        unique_layouts = set()
        all_confs = []
        local_seed = self.seed
        count = 0
        while count < self.num_mazes:
            # carve one wide maze
            local_rng = np.random.RandomState(local_seed)
            local_seed += 1

            # base state: walls everywhere on plane 4
            state = np.zeros(reference_env.state_shape, dtype=bool)
            state[4, :, :] = True

            # carve out the start cell thickly
            init_r, init_c = H - 2, 1
            state[1, init_r, init_c] = True
            for dr in offsets:
                for dc in offsets:
                    rr, cc = init_r + dr, init_c + dc
                    if 0 <= rr < H and 0 <= cc < W:
                        state[4, rr, cc] = False

            # DFS carve corridors
            visited = np.zeros((H, W), dtype=bool)
            visited[init_r, init_c] = True
            stack = [[init_r, init_c]]
            while stack:
                cur_r, cur_c = stack.pop()
                # gather neighbors
                nbrs = []
                if cur_r - 2 > 0:
                    nbrs.append([cur_r - 2, cur_c])
                if cur_r + 2 < H - 1:
                    nbrs.append([cur_r + 2, cur_c])
                if cur_c - 2 > 0:
                    nbrs.append([cur_r, cur_c - 2])
                if cur_c + 2 < W - 1:
                    nbrs.append([cur_r, cur_c + 2])
                local_rng.shuffle(nbrs)

                for nbr_r, nbr_c in nbrs:
                    if visited[nbr_r, nbr_c]:
                        continue
                    visited[nbr_r, nbr_c] = True

                    # carve a thick corridor
                    if cur_r == nbr_r:
                        # horizontal
                        c0, c1 = sorted([cur_c, nbr_c])
                        for dr in offsets:
                            rr = cur_r + dr
                            if not (0 <= rr < H):
                                continue
                            for cc in range(c0, c1 + 1):
                                state[4, rr, cc] = False
                    else:
                        # vertical
                        r0, r1 = sorted([cur_r, nbr_r])
                        for dc in offsets:
                            cc = cur_c + dc
                            if not (0 <= cc < W):
                                continue
                            for rr in range(r0, r1 + 1):
                                state[4, rr, cc] = False

                    stack.append([nbr_r, nbr_c])

            # re-enforce the outer border as walls
            state[4, 0, :] = True
            state[4, H - 1, :] = True
            state[4, :, 0] = True
            state[4, :, W - 1] = True

            # skip duplicate layouts
            layout_key = state[4].tobytes()
            if layout_key in unique_layouts:
                continue
            unique_layouts.add(layout_key)
            count += 1

            # for each open cell, create a marker configuration
            for y in range(1, H - 1):
                for x in range(1, W - 1):
                    if not state[4, y, x] and not (y == init_r and x == init_c):
                        conf = state.copy()
                        conf[5, :, :] = True
                        conf[6, y, x] = True
                        conf[5, y, x] = False
                        all_confs.append(conf)

        return all_confs

    def generate_initial_environment(self, env_args):
        self.done = False
        self.previous_distance = None

        return KarelEnvironment(**env_args)

    def reset_environment(self):
        super().reset_environment()
        self.done = False
        self.previous_distance = None

    def get_reward(self, env: KarelEnvironment):
        terminated = False
        reward = 0.

        karel_pos = env.get_hero_pos()
        marker_pos = np.argwhere(env.state[6, :, :])

        # print(f"** MazeWideSparseAllInit: karel_pos={karel_pos}, marker_pos={marker_pos}")
        LOG(f"** MazeWideSparseAllInit: karel_pos={karel_pos}, marker_pos={marker_pos}")
        
        if len(marker_pos) > 0:
            marker_pos = marker_pos[0]  # (y, x)

        if karel_pos[0] == marker_pos[0] and karel_pos[1] == marker_pos[1]:
            terminated = True
            reward = 1.0
            print("** Agent reached the goal!!!!")
        
        return terminated, reward
    

class MazeAllInit(BaseTask):
    """
    Generate multiple random mazes (via DFS carving), store all open cells for
    each maze, and produce a separate "initial_state" for each possible marker location.
    The agent's start is always the same bottom-left corner [env_height-2, 1],
    but the layout of walls changes from maze to maze.
    """
    def __init__(self, env_args, seed=None, num_mazes=5):
        super().__init__(env_args, seed=seed)
        self.done = False
        self.previous_distance = None
        self.num_mazes = num_mazes
        self.seed = 0 if seed is None else seed # For reproducibility

        # Pre-generate multiple maze layouts and all possible marker positions
        # so we have a big list of states
        self.all_initial_confs = self._generate_all_initial_confs(env_args, self.num_mazes)

    def _carve_maze(self, base_state, env_height, env_width, local_rng):
        """
        Carves out walls from base_state (which starts full of walls),
        using a DFS from the bottom-left corner [env_height-2, 1].
        Mutates base_state in place (plane 4).
        """
        # Helper function to find neighbor cells for DFS
        def get_neighbors(cur_pos):
            neighbor_list = []
            # top
            if cur_pos[0] - 2 > 0:
                neighbor_list.append([cur_pos[0] - 2, cur_pos[1]])
            # bottom
            if cur_pos[0] + 2 < env_height - 1:
                neighbor_list.append([cur_pos[0] + 2, cur_pos[1]])
            # left
            if cur_pos[1] - 2 > 0:
                neighbor_list.append([cur_pos[0], cur_pos[1] - 2])
            # right
            if cur_pos[1] + 2 < env_width - 1:
                neighbor_list.append([cur_pos[0], cur_pos[1] + 2])
            return neighbor_list

        init_pos = [env_height - 2, 1]
        visited = np.zeros((env_height, env_width), dtype=bool)
        visited[init_pos[0], init_pos[1]] = True

        stack = [init_pos]
        while stack:
            cur_pos = stack.pop()
            neighbors = get_neighbors(cur_pos)
            # shuffle with the parent's RNG so we get different mazes each time
            # self.rng.shuffle(neighbors)
            local_rng.shuffle(neighbors)

            for nbr in neighbors:
                if not visited[nbr[0], nbr[1]]:
                    visited[nbr[0], nbr[1]] = True
                    # remove walls in the corridor between cur_pos and nbr
                    base_state[4, (cur_pos[0] + nbr[0]) // 2, (cur_pos[1] + nbr[1]) // 2] = False
                    base_state[4, nbr[0], nbr[1]] = False
                    stack.append(nbr)

    def _generate_all_initial_confs(self, env_args, num_mazes):
        """
        1) For i in [0..num_mazes-1], create a full-wall state, carve a maze with DFS.
        2) For each carved maze, gather all open cells for the marker.
        3) Produce a new initial_state for each open cell (marker) in each maze layout.
        """
        reference_env = KarelEnvironment(**env_args)
        env_height = reference_env.state_shape[1]
        env_width  = reference_env.state_shape[2]

        all_confs = []
        unique_layouts = []

        i = 0  
        while len(unique_layouts) < num_mazes:
            local_seed = self.seed + i
            i += 1  # increment for next loop if needed

            # 1) Make a blank state (all walls) except for the agent position
            state = np.zeros(reference_env.state_shape, dtype=bool)
            # plane 4 => walls
            state[4, :, :] = True

            # Agent at bottom-left corner
            init_pos = [env_height - 2, 1]
            state[1, init_pos[0], init_pos[1]] = True
            state[4, init_pos[0], init_pos[1]] = False  # no wall where agent stands

            # 2) Carve the maze
            local_rng = np.random.RandomState(local_seed)
            self._carve_maze(state, env_height, env_width, local_rng)

            # We convert plane[4,:,:] to bytes so we can compare layout duplicates quickly
            walls_bytes = state[4].tobytes()

            if walls_bytes in unique_layouts:   # means same layout as earlier => skip
                continue
            else:
                unique_layouts.append(walls_bytes)
                print(f"seed: {local_seed}")

            # 3) Find all open cells for markers
            open_positions = []
            for y in range(1, env_height - 1):
                for x in range(1, env_width - 1):
                    if not state[4, y, x]:         # not a wall
                        if [y, x] != init_pos:     # not the agent's cell
                            open_positions.append((y, x))

            # 4) For each open cell => a distinct initial_state
            for (my, mx) in open_positions:
                conf_state = state.copy()
                conf_state[5, :, :] = True
                conf_state[6, my, mx] = True
                conf_state[5, my, mx] = False
                all_confs.append(conf_state)

        return all_confs

    def generate_initial_environment(self, env_args):
        self.done = False
        self.previous_distance = None
        return KarelEnvironment(**env_args)

    def reset_environment(self):
        super().reset_environment()
        self.done = False
        self.previous_distance = None

    def get_reward(self, env: KarelEnvironment):
        # We'll just use the same "dense" Maze logic here:
        terminated = False
        reward = 0.0

        if self.previous_distance is None:
            # Initialize distances once at the start
            karel_pos = env.get_hero_pos()
            marker_pos = np.argwhere(env.state[6, :, :])
            if len(marker_pos) > 0:
                marker_pos = marker_pos[0]  # (y, x)
                self.initial_distance = abs(karel_pos[0] - marker_pos[0]) \
                                      + abs(karel_pos[1] - marker_pos[1])
                self.previous_distance = self.initial_distance
            else:
                self.previous_distance = 1.0

        karel_pos = env.get_hero_pos()
        marker_pos = np.argwhere(env.state[6, :, :])    # no need to extract marker position in generate_initial_environment
        if len(marker_pos) > 0:
            marker_pos = marker_pos[0]  # (y, x)
        else:
            marker_pos = [0, 0]

        current_distance = abs(karel_pos[0] - marker_pos[0]) \
                         + abs(karel_pos[1] - marker_pos[1])

        # "Normalized" difference in distance
        reward = (self.previous_distance - current_distance) / float(self.initial_distance)

        if current_distance == 0:
            terminated = True

        self.previous_distance = current_distance
        return terminated, reward

class MazeSparseAllInit(MazeAllInit):
    """
    Same approach as MazeAllInit but uses a sparse reward:
      - -1 each step unless the agent is on the marker => 0 & done.
    """
    def get_reward(self, env: KarelEnvironment):
        terminated = False
        reward = 0.0

        karel_pos = env.get_hero_pos()
        marker_pos = np.argwhere(env.state[6, :, :])
        if len(marker_pos) > 0:
            marker_pos = marker_pos[0]  # (y, x)

        if karel_pos[0] == marker_pos[0] and karel_pos[1] == marker_pos[1]:
            terminated = True
            reward = 1.0
            # print("** Agent reached the goal!!!!")

        return terminated, reward