import heapq
import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 3
        num_walls = 11
        world.collaborative = True

        world.agents = [Agent() for _ in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = "agent %d" % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15

        world.landmarks = [Landmark() for _ in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.08

        world.walls = [Landmark() for _ in range(num_walls)]
        for i, wall in enumerate(world.walls):
            wall.name = "wall %d" % i
            wall.collide = True
            wall.movable = False
            wall.size = 0.125
            wall.color = np.array([0.2, 0.2, 0.2])

        # 开口向下的连续 U 形障碍：
        # 左右各 3 块竖墙，顶部 5 块横向连接墙。
        # 相邻墙心距离均为 0.25 = 2 * wall.size，保证连续相切、无缝隙。
        self.u_shape_positions = [
            [-0.75, 0.25],
            [-0.75, 0.00],
            [-0.75, -0.25],
            [-0.50, 0.25],
            [-0.25, 0.25],
            [0.00, 0.25],
            [0.25, 0.25],
            [0.50, 0.25],
            [0.75, 0.25],
            [0.75, 0.00],
            [0.75, -0.25],
        ]

        # agent 生成在 U 形内部腔体中，landmark 生成在 U 外部上方区域。
        self.agent_spawn_region = (-0.40, 0.40, -0.36, 0.10)
        self.landmark_spawn_region = (-0.90, 0.90, 0.62, 1.08)
        self.spawn_margin = 0.02


        # Shortest-path PBRS 参数

        # 注意：这些参数只用于势能函数的最短路径距离估计，不改变 observation。
        # grid_step 越小，最短路径越精细，但计算越慢。
        self.path_grid_min = np.array([-1.25, -1.25], dtype=np.float32)
        self.path_grid_max = np.array([1.25, 1.25], dtype=np.float32)
        self.path_grid_step = 0.04

        # 路径规划时额外膨胀墙体的安全距离。
        # 实际障碍膨胀半径 = agent.size + wall.size + path_clearance。
        self.path_clearance = 0.02

        # PBRS 权重
        self.pbrs_weight = 1.0

        # 折扣因子应与训练算法中的 gamma 保持一致。
        self.pbrs_gamma = 0.95

        self.reset_world(world)
        return world


    # Shortest-path potential helpers

    def _grid_shape(self):
        """Return occupancy-grid shape."""
        grid_size = np.ceil(
            (self.path_grid_max - self.path_grid_min) / self.path_grid_step
        ).astype(int) + 1
        return int(grid_size[0]), int(grid_size[1])

    def _world_to_grid(self, pos):
        """
        Convert a continuous 2D position to a grid cell.

        Returns:
            cell: (ix, iy)
            outside_extra: distance from pos to clipped pos if pos is outside grid bounds.
        """
        pos = np.asarray(pos, dtype=np.float32)
        clipped_pos = np.minimum(np.maximum(pos, self.path_grid_min), self.path_grid_max)
        idx = np.rint((clipped_pos - self.path_grid_min) / self.path_grid_step).astype(int)

        nx, ny = self._grid_shape()
        ix = int(np.clip(idx[0], 0, nx - 1))
        iy = int(np.clip(idx[1], 0, ny - 1))

        outside_extra = float(np.linalg.norm(pos - clipped_pos))
        return (ix, iy), outside_extra

    def _grid_to_world(self, cell):
        """Convert a grid cell to a continuous 2D position."""
        ix, iy = cell
        return self.path_grid_min + np.array([ix, iy], dtype=np.float32) * self.path_grid_step

    def _build_occupancy_grid(self, world):
        """
        Build a grid map for path planning.

        True means free cell; False means blocked cell.
        Walls are inflated by agent radius to avoid generating paths that pass through walls.
        """
        nx, ny = self._grid_shape()
        free_grid = np.ones((nx, ny), dtype=bool)

        # 所有 agent 尺寸相同，取第一个 agent 的 size 作为通行半径。
        agent_radius = world.agents[0].size if len(world.agents) > 0 else 0.0

        for ix in range(nx):
            for iy in range(ny):
                pos = self._grid_to_world((ix, iy))
                for wall in world.walls:
                    blocked_radius = agent_radius + wall.size + self.path_clearance
                    if np.linalg.norm(pos - wall.state.p_pos) < blocked_radius:
                        free_grid[ix, iy] = False
                        break

        return free_grid

    def _nearest_free_cell(self, cell, free_grid):
        """
        Return the nearest free cell around a given cell.

        This prevents occasional numerical/grid-discretization failures when a landmark or agent
        is very close to an inflated obstacle boundary.
        """
        ix, iy = cell
        nx, ny = free_grid.shape

        if 0 <= ix < nx and 0 <= iy < ny and free_grid[ix, iy]:
            return cell

        max_radius = max(nx, ny)
        for radius in range(1, max_radius):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    # 只搜索当前方形环边界，避免重复检查内部。
                    if abs(dx) != radius and abs(dy) != radius:
                        continue

                    nx_i = ix + dx
                    ny_i = iy + dy
                    if 0 <= nx_i < nx and 0 <= ny_i < ny and free_grid[nx_i, ny_i]:
                        return (nx_i, ny_i)

        return None

    def _dijkstra_distance_field(self, source_cell, free_grid):
        """
        Compute shortest-path distance from a source cell to all free cells.

        This implementation uses 8-neighbor movement and prevents diagonal corner-cutting.
        """
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

                if not free_grid[nx_i, ny_i]:
                    continue

                # 防止斜向移动时从两个障碍格子的夹角处“穿墙”。
                if dx != 0 and dy != 0:
                    if not (free_grid[ix + dx, iy] and free_grid[ix, iy + dy]):
                        continue

                step_cost = self.path_grid_step * np.sqrt(dx * dx + dy * dy)
                new_dist = current_dist + step_cost

                if new_dist < dist_field[nx_i, ny_i]:
                    dist_field[nx_i, ny_i] = new_dist
                    heapq.heappush(pq, (new_dist, (nx_i, ny_i)))

        return dist_field

    def _build_shortest_path_fields(self, world):
        """
        Precompute a shortest-path distance field for every landmark.

        Since walls and landmarks are fixed within one episode, we only need to rebuild these
        fields in reset_world(), not at every reward call.
        """
        self.path_free_grid = self._build_occupancy_grid(world)
        self.landmark_distance_fields = []

        for landmark in world.landmarks:
            landmark_cell, _ = self._world_to_grid(landmark.state.p_pos)
            distance_field = self._dijkstra_distance_field(landmark_cell, self.path_free_grid)
            self.landmark_distance_fields.append(distance_field)

    def _shortest_path_distance(self, pos, landmark_index, fallback_pos=None):
        """
        Query shortest-path distance from pos to the landmark specified by landmark_index.

        If the field is unavailable or the queried cell is unreachable, fall back to Euclidean
        distance. This makes the reward robust instead of returning inf/nan during training.
        """
        if not hasattr(self, "landmark_distance_fields") or not hasattr(self, "path_free_grid"):
            if fallback_pos is None:
                return 0.0
            return float(np.linalg.norm(pos - fallback_pos))

        cell, outside_extra = self._world_to_grid(pos)
        cell = self._nearest_free_cell(cell, self.path_free_grid)

        if cell is None:
            if fallback_pos is None:
                return 0.0
            return float(np.linalg.norm(pos - fallback_pos))

        dist = float(self.landmark_distance_fields[landmark_index][cell])

        if not np.isfinite(dist):
            if fallback_pos is None:
                return 0.0
            return float(np.linalg.norm(pos - fallback_pos))

        return dist + outside_extra

    def calculate_potential(self, world):
        """
        Shortest-path PBRS potential.

        Original Euclidean potential:
            Phi(s) = -sum_j min_i ||p_i - g_j||_2

        New shortest-path potential:
            Phi_sp(s) = -sum_j min_i d_sp(p_i, g_j; obstacles)
        """
        team_potential = 0.0

        for landmark_index, landmark in enumerate(world.landmarks):
            dists = [
                self._shortest_path_distance(
                    agent.state.p_pos,
                    landmark_index,
                    fallback_pos=landmark.state.p_pos,
                )
                for agent in world.agents
            ]
            team_potential -= min(dists)

        return team_potential


    # Position sampling helpers

    def _is_valid_position(self, pos, radius, world, occupied_positions, occupied_radii):
        for wall in world.walls:
            if np.linalg.norm(pos - wall.state.p_pos) < (radius + wall.size + self.spawn_margin):
                return False

        for other_pos, other_radius in zip(occupied_positions, occupied_radii):
            if np.linalg.norm(pos - other_pos) < (radius + other_radius + self.spawn_margin):
                return False

        return True

    def _sample_position(self, x_range, y_range, radius, world, occupied_positions, occupied_radii, max_trials=4000):
        x_min, x_max = x_range
        y_min, y_max = y_range

        for _ in range(max_trials):
            pos = np.array([
                np.random.uniform(x_min, x_max),
                np.random.uniform(y_min, y_max),
            ])
            if self._is_valid_position(pos, radius, world, occupied_positions, occupied_radii):
                return pos

        # 随机采样失败后，退化为规则网格搜索，避免 reset 偶发失败。
        grid_step = 0.02
        x_candidates = np.arange(x_min, x_max + 1e-9, grid_step)
        y_candidates = np.arange(y_min, y_max + 1e-9, grid_step)

        candidate_positions = [
            np.array([x, y])
            for x in x_candidates
            for y in y_candidates
        ]
        np.random.shuffle(candidate_positions)

        for pos in candidate_positions:
            if self._is_valid_position(pos, radius, world, occupied_positions, occupied_radii):
                return pos

        raise RuntimeError("Failed to sample a valid spawn position. Please enlarge the spawn region.")

    def _sample_group_positions(self, region, radii, world, occupied_positions, occupied_radii, max_restarts=200):
        x_range = (region[0], region[1])
        y_range = (region[2], region[3])

        for _ in range(max_restarts):
            local_positions = []
            local_radii = []
            success = True
            for radius in radii:
                try:
                    pos = self._sample_position(
                        x_range,
                        y_range,
                        radius,
                        world,
                        occupied_positions + local_positions,
                        occupied_radii + local_radii,
                        max_trials=500,
                    )
                except RuntimeError:
                    success = False
                    break
                local_positions.append(pos)
                local_radii.append(radius)
            if success:
                return local_positions

        grid_step = 0.02
        x_candidates = np.arange(region[0], region[1] + 1e-9, grid_step)
        y_candidates = np.arange(region[2], region[3] + 1e-9, grid_step)
        candidate_positions = [
            np.array([x, y])
            for x in x_candidates
            for y in y_candidates
            if self._is_valid_position(np.array([x, y]), radii[0], world, occupied_positions, occupied_radii)
        ]
        np.random.shuffle(candidate_positions)

        def backtrack(idx, chosen_positions):
            if idx == len(radii):
                return [pos.copy() for pos in chosen_positions]

            radius = radii[idx]
            for pos in candidate_positions:
                if not self._is_valid_position(
                    pos,
                    radius,
                    world,
                    occupied_positions + chosen_positions,
                    occupied_radii + radii[:idx],
                ):
                    continue
                chosen_positions.append(pos)
                result = backtrack(idx + 1, chosen_positions)
                if result is not None:
                    return result
                chosen_positions.pop()
            return None

        result = backtrack(0, [])
        if result is None:
            raise RuntimeError("Failed to sample a valid spawn group. Please enlarge the spawn region.")
        return result


    # Scenario API

    def reset_world(self, world):
        for agent in world.agents:
            agent.color = np.array([0.35, 0.35, 0.85])
        for landmark in world.landmarks:
            landmark.color = np.array([0.25, 0.25, 0.25])

        for i, wall in enumerate(world.walls):
            wall.state.p_pos = np.array(self.u_shape_positions[i])
            wall.state.p_vel = np.zeros(world.dim_p)

        occupied_positions = []
        occupied_radii = []

        agent_positions = self._sample_group_positions(
            self.agent_spawn_region,
            [agent.size for agent in world.agents],
            world,
            occupied_positions,
            occupied_radii,
        )

        for agent, pos in zip(world.agents, agent_positions):
            agent.state.p_pos = pos
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            occupied_positions.append(agent.state.p_pos.copy())
            occupied_radii.append(agent.size)

        for landmark in world.landmarks:
            landmark.state.p_pos = self._sample_position(
                (self.landmark_spawn_region[0], self.landmark_spawn_region[1]),
                (self.landmark_spawn_region[2], self.landmark_spawn_region[3]),
                landmark.size,
                world,
                occupied_positions,
                occupied_radii,
            )
            landmark.state.p_vel = np.zeros(world.dim_p)
            occupied_positions.append(landmark.state.p_pos.copy())
            occupied_radii.append(landmark.size)

        # 根据地标与墙体位置预计算最短路径距离场。
        self._build_shortest_path_fields(world)

        # 初始化每个 agent 的 PBRS 前一时刻势能。
        initial_potential = self.calculate_potential(world)
        for agent in world.agents:
            agent.prev_potential = initial_potential

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        min_sp_dists = 0

        for landmark_index, landmark in enumerate(world.landmarks):
            # benchmark 里保留原始欧式距离，便于和 simple-spread 指标兼容。
            euclidean_dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - landmark.state.p_pos)))
                for a in world.agents
            ]
            min_euclidean_dist = min(euclidean_dists)
            min_dists += min_euclidean_dist
            rew -= min_euclidean_dist

            # 计算最短路径距离统计。
            sp_dists = [
                self._shortest_path_distance(
                    a.state.p_pos,
                    landmark_index,
                    fallback_pos=landmark.state.p_pos,
                )
                for a in world.agents
            ]
            min_sp_dists += min(sp_dists)

            if min_euclidean_dist < 0.1:
                occupied_landmarks += 1

        if agent.collide:
            for other in world.agents:
                if other is agent:
                    continue
                if self.is_collision(other, agent):
                    rew -= 1
                    collisions += 1
            for wall in world.walls:
                if self.is_collision(agent, wall):
                    rew -= 1
                    collisions += 1

        # 返回四项统计量，保持 benchmark 接口一致。


        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return dist < dist_min

    def reward(self, agent, world):
        rew = 0

        # 基础奖励保持原来的欧式距离形式。

        for landmark in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - landmark.state.p_pos)))
                for a in world.agents
            ]
            rew -= min(dists)

        if agent.collide:
            for other in world.agents:
                if other is agent:
                    continue
                if self.is_collision(other, agent):
                    rew -= 1
            for wall in world.walls:
                if self.is_collision(agent, wall):
                    rew -= 1

        current_potential = self.calculate_potential(world)

        if not hasattr(agent, "prev_potential"):
            agent.prev_potential = current_potential

        shape_reward = self.pbrs_weight * (
            self.pbrs_gamma * current_potential - agent.prev_potential
        )
        agent.prev_potential = current_potential

        return rew + shape_reward

    def observation(self, agent, world):
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        wall_pos = []
        for wall in world.walls:
            wall_pos.append(wall.state.p_pos - agent.state.p_pos)

        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        # observation 不改变：最短路径距离只用于 PBRS 奖励塑形，不额外喂给 actor。
        return np.concatenate(
            [agent.state.p_vel] + [agent.state.p_pos] + entity_pos + wall_pos + other_pos + comm
        )
