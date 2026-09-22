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

        self.reset_world(world)
        return world

    def calculate_potential(self, world):
        team_potential = 0.0
        for landmark in world.landmarks:
            nearest_agent_dist = min(
                np.linalg.norm(agent.state.p_pos - landmark.state.p_pos)
                for agent in world.agents
            )
            team_potential -= nearest_agent_dist
        return team_potential

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

        initial_potential = self.calculate_potential(world)
        for agent in world.agents:
            agent.prev_potential = initial_potential

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for landmark in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - landmark.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
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

        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return dist < dist_min

    def reward(self, agent, world):
        rew = 0
        for landmark in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - landmark.state.p_pos))) for a in world.agents]
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

        gamma = 0.95
        current_potential = self.calculate_potential(world)

        if not hasattr(agent, "prev_potential"):
            agent.prev_potential = current_potential

        shape_reward = (gamma * current_potential) - agent.prev_potential
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

        return np.concatenate(
            [agent.state.p_vel] + [agent.state.p_pos] + entity_pos + wall_pos + other_pos + comm
        )
