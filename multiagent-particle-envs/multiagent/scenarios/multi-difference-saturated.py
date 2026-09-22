import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        world.dim_c = 2
        world.collaborative = True

        self.reward_mode = "standard_plus_difference_saturated"

        self.num_agents = 3
        self.num_landmarks = 3

        self.agent_size = 0.10
        self.landmark_size = 0.05

        # 墙体由连续相切的圆形实体组成。

        self.wall_size = 0.085
        self.wall_spacing = 0.17

        # sigma 控制覆盖核的衰减宽度。
        self.sigma = 0.35

        # beta 控制覆盖效用的饱和速度。
        # beta 小：多人协同累计覆盖更明显
        # beta 大：更快饱和，更强调“先占到”即可
        self.beta = 2.0

        # 差分奖励权重。
        # 基础奖励 + lambda_D * 饱和覆盖效用差分奖励


        self.diff_weight = 0.1

        # 离散碰撞惩罚参数：
        # 发生一次 agent-agent 碰撞扣 lambda_aa；
        # 发生一次 agent-wall 碰撞扣 lambda_aw。
        self.lambda_aa = 1.0
        self.lambda_aw = 1.0

        self.box_x = 1.15
        self.box_y = 1.15

        # 中心通道略放宽，但仍保持“单通道”几何特征：
        # 当前参数下最窄有效开口约为 0.38。
        # 单个智能体直径为 0.20，可通过；
        # 两个智能体并排至少需要约 0.40，因此不能严格并排同时通过。
        self.channel_half_height = 0.25

        self.left_spawn_region = (-0.95, -0.40, -0.85, 0.85)
        self.right_spawn_region = (0.40, 0.95, -0.85, 0.85)
        self.spawn_margin = 0.03

        self.wall_positions = self._build_wall_positions()

        world.agents = [Agent() for _ in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = "agent_%d" % i
            agent.index = i
            agent.collide = True
            agent.silent = True
            agent.size = self.agent_size

        world.landmarks = [Landmark() for _ in range(self.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "target_%d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = self.landmark_size

        world.walls = [Landmark() for _ in range(len(self.wall_positions))]
        for i, wall in enumerate(world.walls):
            wall.name = "wall_%d" % i
            wall.collide = True
            wall.movable = False
            wall.size = self.wall_size
            wall.color = np.array([0.25, 0.25, 0.25])

        self.reset_world(world)
        return world

    def _build_wall_positions(self):
        positions = []

        xs = np.arange(-self.box_x, self.box_x + 1e-9, self.wall_spacing)
        ys = np.arange(
            -self.box_y + self.wall_spacing,
            self.box_y - self.wall_spacing + 1e-9,
            self.wall_spacing,
        )

        for x in xs:
            positions.append([x, -self.box_y])
            positions.append([x, self.box_y])

        for y in ys:
            positions.append([-self.box_x, y])
            positions.append([self.box_x, y])

        lower_segment = np.arange(
            -self.box_y + self.wall_spacing,
            -self.channel_half_height + 1e-9,
            self.wall_spacing,
        )
        upper_segment = np.arange(
            self.channel_half_height,
            self.box_y - self.wall_spacing + 1e-9,
            self.wall_spacing,
        )

        for y in lower_segment:
            positions.append([0.0, y])
        for y in upper_segment:
            positions.append([0.0, y])

        return [np.array(pos) for pos in positions]

    def _sample_position(self, region, radius, world, occupied_positions, occupied_radii, max_trials=4000):
        x_min, x_max, y_min, y_max = region

        for _ in range(max_trials):
            pos = np.array([
                np.random.uniform(x_min, x_max),
                np.random.uniform(y_min, y_max),
            ])

            valid = True

            for wall in world.walls:
                if np.linalg.norm(pos - wall.state.p_pos) < (radius + wall.size + self.spawn_margin):
                    valid = False
                    break
            if not valid:
                continue

            for other_pos, other_radius in zip(occupied_positions, occupied_radii):
                if np.linalg.norm(pos - other_pos) < (radius + other_radius + self.spawn_margin):
                    valid = False
                    break
            if not valid:
                continue

            return pos

        raise RuntimeError("Failed to sample a valid spawn position in multi-difference-saturated.")

    def reset_world(self, world):
        for agent in world.agents:
            agent.color = np.array([0.35, 0.35, 0.85])
        for landmark in world.landmarks:
            landmark.color = np.array([0.15, 0.85, 0.15])
        for wall in world.walls:
            wall.color = np.array([0.25, 0.25, 0.25])

        for wall, wall_pos in zip(world.walls, self.wall_positions):
            wall.state.p_pos = wall_pos.copy()
            wall.state.p_vel = np.zeros(world.dim_p)

        agents_on_left = np.random.rand() < 0.5
        agent_region = self.left_spawn_region if agents_on_left else self.right_spawn_region
        landmark_region = self.right_spawn_region if agents_on_left else self.left_spawn_region

        occupied_positions = []
        occupied_radii = []

        for agent in world.agents:
            agent.state.p_pos = self._sample_position(
                agent_region,
                agent.size,
                world,
                occupied_positions,
                occupied_radii,
            )
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            occupied_positions.append(agent.state.p_pos.copy())
            occupied_radii.append(agent.size)

        for landmark in world.landmarks:
            landmark.state.p_pos = self._sample_position(
                landmark_region,
                landmark.size,
                world,
                occupied_positions,
                occupied_radii,
            )
            landmark.state.p_vel = np.zeros(world.dim_p)
            occupied_positions.append(landmark.state.p_pos.copy())
            occupied_radii.append(landmark.size)

        self._cached_metrics = None

    def _state_signature(self, world):
        signature = []
        for agent in world.agents:
            signature.extend(agent.state.p_pos.tolist())
        for landmark in world.landmarks:
            signature.extend(landmark.state.p_pos.tolist())
        return tuple(signature)

    def is_collision(self, entity_a, entity_b):
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = entity_a.size + entity_b.size
        return dist < dist_min

    def _agents_except(self, world, excluded_agent=None):
        if excluded_agent is None:
            return world.agents
        return [agent for agent in world.agents if agent is not excluded_agent]

    def _distance(self, agent, landmark):
        return np.sqrt(np.sum(np.square(agent.state.p_pos - landmark.state.p_pos)))

    def _kernel(self, dist):
        return np.exp(-(dist ** 2) / (self.sigma ** 2))

    def _global_agent_agent_collisions(self, agents):
        collisions = 0
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                if self.is_collision(agents[i], agents[j]):
                    collisions += 1
        return collisions

    def _global_agent_wall_collisions(self, agents, walls):
        collisions = 0
        for agent in agents:
            for wall in walls:
                if self.is_collision(agent, wall):
                    collisions += 1
        return collisions

    def _global_standard_utility(self, world, excluded_agent=None):
        agents = self._agents_except(world, excluded_agent)
        if len(agents) == 0:
            return 0.0

        utility = 0.0
        for landmark in world.landmarks:
            dists = [self._distance(agent, landmark) for agent in agents]
            utility -= min(dists)

        aa_collisions = self._global_agent_agent_collisions(agents)
        aw_collisions = self._global_agent_wall_collisions(agents, world.walls)

        utility -= self.lambda_aa * aa_collisions
        utility -= self.lambda_aw * aw_collisions

        return utility

    def _global_saturated_utility(self, world, excluded_agent=None):
        agents = self._agents_except(world, excluded_agent)
        if len(agents) == 0:
            return 0.0

        utility = 0.0
        for landmark in world.landmarks:
            total_coverage = 0.0
            for agent in agents:
                dist = self._distance(agent, landmark)
                total_coverage += self._kernel(dist)
            utility += 1.0 - np.exp(-self.beta * total_coverage)

        aa_collisions = self._global_agent_agent_collisions(agents)
        aw_collisions = self._global_agent_wall_collisions(agents, world.walls)

        utility -= self.lambda_aa * aa_collisions
        utility -= self.lambda_aw * aw_collisions

        return utility

    def _difference_saturated_utility(self, agent, world):
        return self._global_saturated_utility(world) - self._global_saturated_utility(
            world,
            excluded_agent=agent,
        )

    def _build_step_metrics(self, world):
        signature = self._state_signature(world)
        cached = getattr(self, "_cached_metrics", None)
        if cached is not None and cached["signature"] == signature:
            return cached

        # 基础共享奖励：保持与 shared-standard 对照组一致，提供直接的目标接近信号。
        global_standard = self._global_standard_utility(world)

        # 饱和覆盖效用差分奖励：作为辅助项刻画个体边际贡献。
        global_saturated = self._global_saturated_utility(world)
        difference_rewards = [
            global_saturated - self._global_saturated_utility(world, excluded_agent=other)
            for other in world.agents
        ]

        # 合成奖励：基础奖励 + 加权差分奖励
        # 注意：基础奖励为团队共享标量，差分奖励为 agent-specific 标量。
        combined_rewards = [
            global_standard + self.diff_weight * diff_reward
            for diff_reward in difference_rewards
        ]

        collisions = [0] * len(world.agents)
        if world.agents and world.agents[0].collide:
            for i, agent_i in enumerate(world.agents):
                for j in range(i + 1, len(world.agents)):
                    agent_j = world.agents[j]
                    if self.is_collision(agent_i, agent_j):
                        collisions[i] += 1
                        collisions[j] += 1
                for wall in world.walls:
                    if self.is_collision(agent_i, wall):
                        collisions[i] += 1

        occupied_landmarks = 0
        min_dists = 0.0
        for landmark in world.landmarks:
            dists = [self._distance(agent, landmark) for agent in world.agents]
            min_dist = min(dists)
            min_dists += min_dist
            if min_dist < (self.agent_size + self.landmark_size):
                occupied_landmarks += 1

        metrics = {
            "signature": signature,
            "shared_reward": global_standard,
            "saturated_global": global_saturated,
            "difference_rewards": difference_rewards,
            "combined_rewards": combined_rewards,
            "collisions": collisions,
            "min_dists": min_dists,
            "occupied_landmarks": occupied_landmarks,
        }
        self._cached_metrics = metrics
        return metrics

    def benchmark_data(self, agent, world):
        metrics = self._build_step_metrics(world)
        return (
            metrics["combined_rewards"][agent.index],
            metrics["collisions"][agent.index],
            metrics["min_dists"],
            metrics["occupied_landmarks"],
        )

    def reward(self, agent, world):
        metrics = self._build_step_metrics(world)
        return metrics["combined_rewards"][agent.index]

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
            [agent.state.p_vel]
            + [agent.state.p_pos]
            + entity_pos
            + wall_pos
            + other_pos
            + comm
        )
