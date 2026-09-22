import heapq
import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 3
        world.collaborative = True

        # 开放场景不设置墙体。


        world.walls = []

        # 最短路径势能参数
        self.gamma = 0.95
        self.pbrs_weight = 1.0
        self.path_grid_min = np.array([-1.25, -1.25], dtype=np.float32)
        self.path_grid_max = np.array([1.25, 1.25], dtype=np.float32)
        self.path_grid_step = 0.04
        self.path_clearance = 0.02

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False

        # make initial conditions
        self.reset_world(world)
        return world


    # Shortest-path potential utilities

    def _get_obstacles(self, world):
        """Return static obstacles used by shortest-path potential.

        In r1.0, world.walls is empty. If walls are added later, this method
        automatically uses them as non-traversable obstacles.
        """
        return getattr(world, 'walls', [])

    def _grid_shape(self):
        grid_size = np.floor(
            (self.path_grid_max - self.path_grid_min) / self.path_grid_step
        ).astype(int) + 1
        return int(grid_size[0]), int(grid_size[1])

    def _world_to_grid(self, pos):
        pos = np.asarray(pos, dtype=np.float32)
        clipped_pos = np.minimum(np.maximum(pos, self.path_grid_min), self.path_grid_max)
        idx = np.rint((clipped_pos - self.path_grid_min) / self.path_grid_step).astype(int)

        nx, ny = self._grid_shape()
        ix = int(np.clip(idx[0], 0, nx - 1))
        iy = int(np.clip(idx[1], 0, ny - 1))

        # If an agent moves outside the grid range, add this extra distance as a
        # conservative fallback instead of returning inf.
        outside_extra = float(np.linalg.norm(pos - clipped_pos))
        return (ix, iy), outside_extra

    def _grid_to_world(self, cell):
        ix, iy = cell
        return self.path_grid_min + np.array([ix, iy], dtype=np.float32) * self.path_grid_step

    def _build_occupancy_grid(self, world):
        """Build a boolean traversability grid.

        True means free; False means occupied by wall/obstacle.
        """
        nx, ny = self._grid_shape()
        free_grid = np.ones((nx, ny), dtype=bool)

        obstacles = self._get_obstacles(world)
        if len(obstacles) == 0:
            return free_grid

        # Use the largest agent radius for conservative obstacle inflation.
        agent_radius = max(agent.size for agent in world.agents) if world.agents else 0.0

        for ix in range(nx):
            for iy in range(ny):
                pos = self._grid_to_world((ix, iy))

                for obs in obstacles:
                    if not getattr(obs, 'collide', True):
                        continue
                    obs_pos = obs.state.p_pos
                    obs_size = getattr(obs, 'size', 0.0)
                    min_dist = agent_radius + obs_size + self.path_clearance
                    if np.linalg.norm(pos - obs_pos) < min_dist:
                        free_grid[ix, iy] = False
                        break

        return free_grid

    def _nearest_free_cell(self, cell, free_grid):
        ix, iy = cell
        nx, ny = free_grid.shape

        if 0 <= ix < nx and 0 <= iy < ny and free_grid[ix, iy]:
            return cell

        max_radius = max(nx, ny)
        for radius in range(1, max_radius):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius:
                        continue

                    nx_i = ix + dx
                    ny_i = iy + dy
                    if 0 <= nx_i < nx and 0 <= ny_i < ny and free_grid[nx_i, ny_i]:
                        return (nx_i, ny_i)

        return None

    def _can_move_between(self, current_cell, next_cell, free_grid):
        """Check whether a grid move is valid.

        For diagonal moves, prevent corner cutting through obstacle corners.
        """
        x0, y0 = current_cell
        x1, y1 = next_cell

        if not free_grid[x1, y1]:
            return False

        dx = x1 - x0
        dy = y1 - y0

        if abs(dx) == 1 and abs(dy) == 1:
            if not free_grid[x0 + dx, y0]:
                return False
            if not free_grid[x0, y0 + dy]:
                return False

        return True

    def _dijkstra_distance_field(self, source_cell, free_grid):
        """Compute shortest-path distance from source to every free cell."""
        nx, ny = free_grid.shape
        dist_field = np.full((nx, ny), np.inf, dtype=np.float32)

        source_cell = self._nearest_free_cell(source_cell, free_grid)
        if source_cell is None:
            return dist_field

        dist_field[source_cell] = 0.0
        pq = [(0.0, source_cell)]

        neighbor_dirs = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]

        while pq:
            current_dist, (ix, iy) = heapq.heappop(pq)

            if current_dist > dist_field[ix, iy]:
                continue

            for dx, dy in neighbor_dirs:
                nx_i = ix + dx
                ny_i = iy + dy

                if not (0 <= nx_i < nx and 0 <= ny_i < ny):
                    continue

                if not self._can_move_between((ix, iy), (nx_i, ny_i), free_grid):
                    continue

                step_cost = self.path_grid_step * np.sqrt(dx * dx + dy * dy)
                new_dist = current_dist + step_cost

                if new_dist < dist_field[nx_i, ny_i]:
                    dist_field[nx_i, ny_i] = new_dist
                    heapq.heappush(pq, (new_dist, (nx_i, ny_i)))

        return dist_field

    def _build_shortest_path_fields(self, world):
        """Precompute one distance field for each landmark.

        Since landmarks and walls are static during an episode, computing once
        in reset_world() is much faster than running path search every reward call.
        """
        self.path_free_grid = self._build_occupancy_grid(world)
        self.landmark_distance_fields = []

        for landmark in world.landmarks:
            landmark_cell, _ = self._world_to_grid(landmark.state.p_pos)
            field = self._dijkstra_distance_field(landmark_cell, self.path_free_grid)
            self.landmark_distance_fields.append(field)

    def _shortest_path_distance(self, pos, landmark_index, landmark_pos=None):
        """Query shortest-path distance from pos to one landmark.

        If the distance field is unavailable or the point is unreachable due to
        grid discretization, fall back to Euclidean distance to avoid inf/nan in
        training rewards.
        """
        if not hasattr(self, 'landmark_distance_fields'):
            if landmark_pos is None:
                return np.inf
            return float(np.linalg.norm(pos - landmark_pos))

        cell, outside_extra = self._world_to_grid(pos)
        cell = self._nearest_free_cell(cell, self.path_free_grid)

        if cell is None:
            if landmark_pos is None:
                return np.inf
            return float(np.linalg.norm(pos - landmark_pos))

        dist = self.landmark_distance_fields[landmark_index][cell]

        if not np.isfinite(dist):
            if landmark_pos is None:
                return np.inf
            return float(np.linalg.norm(pos - landmark_pos))

        return float(dist + outside_extra)

    def calculate_potential(self, world):
        """Shortest-path PBRS potential.

        Original potential:
            Phi(s) = - sum_j min_i ||p_i - g_j||_2

        New potential:
            Phi_sp(s) = - sum_j min_i d_sp(p_i, g_j; obstacles)

        In the original pbrs-r1.0 environment there are no walls, so the shortest
        path is approximately the Euclidean path. If walls are later added to
        world.walls, this function will reflect obstacle-aware path length.
        """
        if not hasattr(self, 'landmark_distance_fields'):
            self._build_shortest_path_fields(world)

        team_potential = 0.0
        for landmark_index, landmark in enumerate(world.landmarks):
            dists = [
                self._shortest_path_distance(
                    agent.state.p_pos,
                    landmark_index,
                    landmark.state.p_pos,
                )
                for agent in world.agents
            ]
            team_potential -= min(dists)

        return team_potential


    # Environment callbacks

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])

        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        # Build shortest-path fields after landmark and obstacle positions are ready.
        self._build_shortest_path_fields(world)

        # Initialize previous potential for every agent.
        initial_potential = self.calculate_potential(world)
        for agent in world.agents:
            agent.prev_potential = initial_potential

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0

        for l in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                for a in world.agents
            ]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1

        if agent.collide:
            for a in world.agents:
                if a is agent:
                    continue
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1

            for wall in self._get_obstacles(world):
                if self.is_collision(agent, wall):
                    rew -= 1
                    collisions += 1

        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        rew = 0

        # Base reward remains Euclidean, matching simple_spread style.
        # Only the PBRS potential is changed to shortest-path distance.
        for l in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                for a in world.agents
            ]
            rew -= min(dists)

        if agent.collide:
            for a in world.agents:
                if a is agent:
                    continue
                if self.is_collision(a, agent):
                    rew -= 1

            for wall in self._get_obstacles(world):
                if self.is_collision(agent, wall):
                    rew -= 1

        current_potential = self.calculate_potential(world)

        # Safety guard for cases where reset_world was bypassed.
        if not hasattr(agent, 'prev_potential'):
            agent.prev_potential = current_potential

        shape_reward = self.pbrs_weight * (
            self.gamma * current_potential - agent.prev_potential
        )
        agent.prev_potential = current_potential

        return rew + shape_reward

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # entity colors are kept for compatibility with the original file,
        # although they are not used in the returned observation.
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)

        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        return np.concatenate(
            [agent.state.p_vel]
            + [agent.state.p_pos]
            + entity_pos
            + other_pos
            + comm
        )
