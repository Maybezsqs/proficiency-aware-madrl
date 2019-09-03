import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 4
        #world.damping = 1
        num_good_agents = 2
        num_adversaries = 2
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 9
        num_food = 2
        num_forests = 2
        num_lawns = 7 #8
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.leader = True if i == 0 else False
            agent.silent = True if i > 0 else False
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.075 if agent.adversary else 0.045
            agent.accel = 3.0 if agent.adversary else 4.0
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
        # add landmarks
        # Change landmarks to buildings and lawns in KSU map
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        world.food = [Landmark() for i in range(num_food)]
        for i, landmark in enumerate(world.food):
            landmark.name = 'food %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.03
            landmark.boundary = False
        world.forests = [Landmark() for i in range(num_forests)]
        for i, landmark in enumerate(world.forests):
            landmark.name = 'forest %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.1
            landmark.boundary = False
        world.lawns = [Landmark() for i in range(num_lawns)]
        for i, landmark in enumerate(world.lawns):
            landmark.name = 'lawn %d' % i
            # TODO what does collide property mean for all entities?
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.1
            # TODO what does boundary property mean here?
            landmark.boundary = False

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
        # TODO currently without the lawn of circle shape
        world.lawn_coordinations = [[(-19.0,-14.0),
                                                (2.0,-14.0),
                                                (-4.5,-24.0),
                                                (-11.0,-24.0)],
                                    [(-19.5,-20.0),
                                                (-16.5,-25.5),
                                                (-16.5,-30.5),
                                                (-19.5,-37.0)],
                                    [(-1.0,-24.5),
                                                (4.5,-18.5),
                                                (4.5,-36.5),
                                                (-1.0,-30.5)],
                                    [(-11.0,-32.5),
                                                (-4.5,-32.5),
                                                (2.0,-42.0),
                                                (-19.0,-42.0)],
                                    [(-11.5,-1.0),
                                                (-8.0,-1.0),
                                                (-8.0,-4.5),
                                                (-11.5,-4.5)],
                                    [(14.5,11.0),
                                                (27.0,8.0),
                                                (15.5,-4.5)],
                                    [(28.0,2.0),
                                                (27.5,-11.0),
                                                (17.5,-10.0)]]
        # Landmarks in world: landmarks+food+forests
        world.landmarks += world.food
        world.landmarks += world.forests
        world.landmarks += world.lawns
        # TODO the line below:
        #world.landmarks += self.set_boundaries(world)  # world boundaries now penalized with negative reward
        # make initial conditions
        self.reset_world(world)
        return world

    def set_boundaries(self, world):
        boundary_list = []
        landmark_size = 1
        edge = 1 + landmark_size
        num_landmarks = int(edge * 2 / landmark_size)
        for x_pos in [-edge, edge]:
            for i in range(num_landmarks):
                l = Landmark()
                l.state.p_pos = np.array([x_pos, -1 + i * landmark_size])
                boundary_list.append(l)

        for y_pos in [-edge, edge]:
            for i in range(num_landmarks):
                l = Landmark()
                l.state.p_pos = np.array([-1 + i * landmark_size, y_pos])
                boundary_list.append(l)

        for i, l in enumerate(boundary_list):
            l.name = 'boundary %d' % i
            l.collide = True
            l.movable = False
            l.boundary = True
            l.color = np.array([0.75, 0.75, 0.75])
            l.size = landmark_size
            l.state.p_vel = np.zeros(world.dim_p)

        return boundary_list


    def reset_world(self, world):
        # random properties for agents
        # grey:building:[0.50, 0.60, 0.60]
        # dark green:forest:[0.25, 0.50, 0.50]
        # shadow green:lawn:[0.6, 0.9, 0.6]
        # blue:good food:[0.15, 0.15, 0.65]
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.45, 0.95, 0.45]) if not agent.adversary else np.array([0.95, 0.45, 0.45])
            agent.color -= np.array([0.3, 0.3, 0.3]) if agent.leader else np.array([0, 0, 0])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.50, 0.55, 0.55])
        for i, landmark in enumerate(world.food):
            landmark.color = np.array([0.15, 0.15, 0.65])
        for i, landmark in enumerate(world.forests):
            landmark.color = np.array([0.25, 0.50, 0.50])
        for i, landmark in enumerate(world.lawns):
            landmark.color = np.array([0.6, 0.9, 0.6])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, landmark in enumerate(world.food):
            landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, landmark in enumerate(world.forests):
            landmark.state.p_pos = np.random.uniform(-0.9, -0.8, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        '''
        for i, landmark in enumerate(world.lawns):
            landmark.state.p_pos = 
        '''


    def benchmark_data(self, agent, world):
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
        #boundary_reward = -10 if self.outside_boundary(agent) else 0
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def outside_boundary(self, agent):
        if agent.state.p_pos[0] > 1 or agent.state.p_pos[0] < -1 or agent.state.p_pos[1] > 1 or agent.state.p_pos[1] < -1:
            return True
        else:
            return False


    def agent_reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        if shape:
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 5
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)  # 1 + (x - 1) * (x - 1)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= 2 * bound(x)

        for food in world.food:
            if self.is_collision(agent, food):
                rew += 2
        rew += 0.05 * min([np.sqrt(np.sum(np.square(food.state.p_pos - agent.state.p_pos))) for food in world.food])

        return rew

    def adversary_reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        rew = 0
        shape = True
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:
            rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) for a in agents])
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 5
        return rew

    # Observation settings below is very important!! Keep in mind!! Need more modifications!!


    def observation2(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        food_pos = []
        for entity in world.food:
            if not entity.boundary:
                food_pos.append(entity.state.p_pos - agent.state.p_pos)
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

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # Self in forest detection
        in_forest = [np.array([-1]), np.array([-1])]
        inf1 = False
        inf2 = False
        if self.is_collision(agent, world.forests[0]):
            in_forest[0] = np.array([1])
            inf1= True
        if self.is_collision(agent, world.forests[1]):
            in_forest[1] = np.array([1])
            inf2 = True

        food_pos = []
        for entity in world.food:
            if not entity.boundary:
                food_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []   
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            oth_f1 = self.is_collision(other, world.forests[0])
            oth_f2 = self.is_collision(other, world.forests[1])
            ## Settings about same forest vision block
            ## should be in the same forest or both not in the forest or is the leader itself than can have the vision
            if (inf1 and oth_f1) or (inf2 and oth_f2) or (not inf1 and not oth_f1 and not inf2 and not oth_f2) or agent.leader:  #without forest vis
                other_pos.append(other.state.p_pos - agent.state.p_pos)
                if not other.adversary:
                    other_vel.append(other.state.p_vel)
            else:
                other_pos.append([0, 0])
                if not other.adversary:
                    other_vel.append([0, 0])

        # to tell the pred when the prey are in the forest
        prey_forest = []
        ga = self.good_agents(world)
        for a in ga:
            if any([self.is_collision(a, f) for f in world.forests]):
                prey_forest.append(np.array([1]))
            else:
                prey_forest.append(np.array([-1]))
        # to tell leader when pred are in forest
        prey_forest_lead = []
        for f in world.forests:
            if any([self.is_collision(a, f) for a in ga]):
                prey_forest_lead.append(np.array([1]))
            else:
                prey_forest_lead.append(np.array([-1]))

        comm = [world.agents[0].state.c]

        if agent.adversary and not agent.leader:
            return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel + in_forest + comm)
        if agent.leader:
            return np.concatenate(
                [agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel + in_forest + comm)
        else:
            return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + in_forest + other_vel)