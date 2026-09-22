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

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15

            # 保存上一时刻动作，用于计算动作变化惩罚。
            agent.prev_action = np.zeros(world.dim_p)

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
            # 每个回合重置动作历史。
            agent.prev_action = np.zeros(world.dim_p)

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

    def benchmark_data(self, agent, world):

        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min_dists
            if min_dists < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # 覆盖距离奖励与碰撞惩罚
        rew = 0
        # 基于到地标点的距离
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists) # 协作性：所有智能体共同覆盖地标

        # 碰撞惩罚
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1 # 碰撞惩罚

        # 自适应动作平滑惩罚


        max_lambda = 0.005    # 平滑惩罚的基础权重
        d_safe = 0.4         # 安全距离阈值

        # 当前智能体到其他智能体的最近距离
        dists_to_others = []
        for other in world.agents:
            if other is not agent: # 排除自己
                dist = np.linalg.norm(agent.state.p_pos - other.state.p_pos)
                dists_to_others.append(dist)

        # 单智能体情形使用无穷大距离。
        d_min = min(dists_to_others) if dists_to_others else float('inf')

        # 当前物理动作
        current_action = agent.action.u if agent.action.u is not None else np.zeros(world.dim_p)

        # 相邻时刻动作差的平方范数
        action_diff_sq = np.sum(np.square(current_action - agent.prev_action))

        # 距离自适应权重
        # 距离增大时权重趋近 1，距离趋近 0 时权重趋近 0。
        adaptive_weight = 1.0 - np.exp(-d_min / d_safe)

        # 计算平滑惩罚
        smooth_penalty = -max_lambda * adaptive_weight * action_diff_sq

        # 合成奖励
        rew += smooth_penalty

        # 更新动作历史。
        # 复制数组，避免后续原地更新改变历史动作。
        agent.prev_action = np.copy(current_action)

        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        return np.concatenate([agent.state.p_vel] +
                             [agent.state.p_pos] +
                             entity_pos +
                             other_pos +
                             comm)
