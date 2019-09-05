import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_good_agents = 1
        num_adversaries = 3
        num_agents = num_adversaries + num_good_agents

        # For KSU
        world.building_coordinations = [[(-0.126585027793863,16.8195412372661),
                                                (-1.26360571382314,26.9366491540346),
                                                (-15.8471949722061,25.2976587627339),
                                                (-14.7101742861769,15.1805508459654)],
                                    [(-23.0767794626376,16.8481382394501),
                                                (-15.1990122585185,17.6638842108194),
                                                (-16.2936205373624,28.2346617605499),
                                                (-24.1713877414815,27.4189157891806)],
                                    [(-24.6491019813129,-2.8185334334924),
                                                (-14.101147340091,-2.84946701921701),
                                                (-14.0468980186871,15.6488534334924),
                                                (-24.594852659909,15.679787019217)],
                                    [(-21.5000842110591,-9.44961399824389),
                                                (-14.0429792391866,-9.28372083691639),
                                                (-14.1765157889409,-3.28108600175611),
                                                (-21.6336207608134,-3.44697916308361)],
                                    [(3.94535013848411,-3.36623865637194),
                                                (14.493304779706,-3.39717224209654),
                                                (14.5540698615159,17.3229386563719),
                                                (4.00611522029401,17.3538722420965)],
                                    [(8.04858173555005,-30.4958667788278),
                                                (12.4781226874434,-30.5088571252425),
                                                (12.53001826445,-12.8131332211722),
                                                (8.10047731255663,-12.8001428747575)],
                                    [(16.1178588413269,-32.2262223173032),
                                                (20.5473997932202,-32.2392126637179),
                                                (20.5917411586731,-17.1193776826968),
                                                (16.1622002067798,-17.1063873362821)],
                                    [(20.5041475657755,-32.1942704687495),
                                                (24.9336885176689,-32.2072608151642),
                                                (24.9464524342245,-27.8549295312505),
                                                (20.5169114823311,-27.8419391848358)],
                                    [(19.5743191710178,-16.0103447371203),
                                                (25.002585828147,-16.0262640084564),
                                                (25.0170808289822,-11.0836552628797),
                                                (19.588814171853,-11.0677359915436)]]

        num_landmarks = len(world.building_coordinations) #2
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 4.0
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world


    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)


    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = False
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)
