import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

import multiagent.scenarios.ksu_map as ksu
import random
import time

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        # Set communication state dimension
        # State: set and changed by action
        world.dim_c = 4 # default 0
        #world.damping = 1

        self.half_dia = np.sqrt(np.sum(np.square(np.array([25.0,41.25]))))
        self.num_good_agents = 1
        self.num_adversaries = 3
        self.num_uav = 2
        self.num_cooperator = 2
        num_agents = self.num_adversaries + self.num_good_agents

        world.building_coordinations = ksu.building_coordinations
        world.forest_coordinations = ksu.forest_coordinations
        world.lawn_coordinations = ksu.lawn_coordinations # TODO currently without the lawn of circle shape
        world.food_coordinations = ksu.food_coordinations 

        num_landmarks = len(world.building_coordinations)
        num_forests = len(world.forest_coordinations)
        num_lawns = len(world.lawn_coordinations)
        num_food = len(world.food_coordinations)

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.leader = True if i == 0 else False
            agent.silent = True if i > 0 else False # Only the leader(agent 0) can speak and others are all silent
            
            # Adversaries special
            agent.adversary = True if i < self.num_adversaries else False # This is very important, first adversaries, then good agents
            agent.ugv = False if i < self.num_uav else True
            agent.size = 0.98 / 2 / 25.0 # 0.045 if agent.adversary else 0.025 # Please refer to reality
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.accel = 1.0 if agent.adversary else 2.0 # Please refer to reality
            agent.max_speed = 0.7 if agent.ugv else 1.5 # Please refer to reality
            agent.max_speed += 0.7 if not agent.adversary else 0 # Please refer to reality
            agent.initial_mass = 1.0 if agent.ugv else 0.7 # Please refer to reality

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks): # buildings
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            coor = ksu.building_coordinations[i]
            center_x, center_y = 0.0, 0.0
            for j in range(len(coor)):
                center_x += coor[j][0]
                center_y += coor[j][1]
            center = np.array([center_x / 4.0, center_y / 4.0]) # physical big range
            landmark.state.p_pos = np.array([center[0] / 25.0, center[1] / 41.25])
            max_sqrsum = max(np.sum(np.square(c - center)) for c in coor)
            landmark.size = np.sqrt(max_sqrsum) / self.half_dia
            landmark.boundary = False
        world.food = [Landmark() for i in range(num_food)]
        for i, landmark in enumerate(world.food):
            landmark.name = 'food %d' % i
            landmark.collide = False
            landmark.size = 0.015 # The target of red ugv
            landmark.boundary = False
            landmark.state.p_pos = np.array([ksu.food_coordinations[i][0] / 25.0, ksu.food_coordinations[i][1] / 41.25])
        # triangle
        world.forests = [Landmark() for i in range(num_forests)]
        for i, landmark in enumerate(world.forests):
            landmark.name = 'forest %d' % i
            landmark.collide = False
            coor = ksu.forest_coordinations[i]
            center_x, center_y, landmark.size = self.getCircle(coor[0], coor[1], coor[2])
            landmark.size /= self.half_dia
            landmark.state.p_pos = np.array([center_x / 25.0, center_y / 41.25])
            landmark.boundary = False
        world.lawns = [Landmark() for i in range(num_lawns)]
        for i, landmark in enumerate(world.lawns):
            landmark.name = 'lawn %d' % i
            landmark.collide = False
            # TODO should be the same with forests and buildings
            landmark.size = None
            # TODO what does boundary property mean here?
            landmark.boundary = False

        # Landmarks in world: landmarks + forests + lawn + food
        world.landmarks += world.forests
        world.landmarks += world.lawns 
        world.landmarks += world.food
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
            agent.color = np.array([0.95, 0.45, 0.45]) if not agent.adversary else np.array([0.45, 0.95, 0.45])
            #agent.color -= np.array([0.3, 0.3, 0.3]) if agent.leader else np.array([0, 0, 0])
            agent.color -= np.array([0.3, 0.3, 0.3]) if agent.ugv and agent.adversary else np.array([0, 0, 0])
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
        # For experiment setting, I give three fixed positions for agents to start
        for i, agent in enumerate(world.agents):
            #agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            # TODO This 1 is expected to be changed as tests and experiments
            if i < self.num_adversaries:
                agent.state.p_pos = np.array([ksu.red_chaser_init_pos[i][0] / 25.0, ksu.red_chaser_init_pos[i][1] / 41.25])
            else:
                random.seed(time.time() * 10000 / 403.712 - 100)
                #randomInit = random.sample(ksu.green_escaper_init_pos,1)
                #agent.state.p_pos = np.array([randomInit[0][0] / 25.0, randomInit[0][1] / 41.25])
                randomInit = random.sample(ksu.escaper_starter_range,1)
                secure_random = random.SystemRandom()
                escaper_x = secure_random.uniform(randomInit[0][0][0],randomInit[0][0][1])
                escaper_y = secure_random.uniform(randomInit[0][1][0],randomInit[0][1][1])
                agent.state.p_pos = np.array([escaper_x / 25.0, escaper_y / 41.25])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, lawns in enumerate(world.lawns):
            lawns.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
            lawns.state.p_vel = np.zeros(world.dim_p)


    def benchmark_data(self, agent, world):
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent, 1.0):
                    collisions += 1
            return collisions
        else:
            return 0


    def getCircle(self, p1, p2, p3):
        x21 = p2[0] - p1[0]
        y21 = p2[1] - p1[1]
        x32 = p3[0] - p2[0]
        y32 = p3[1] - p2[1]
        # three colinear
        if (x21 * y32 - x32 * y21 == 0):
            return None
        xy21 = p2[0] * p2[0] - p1[0] * p1[0] + p2[1] * p2[1] - p1[1] * p1[1]
        xy32 = p3[0] * p3[0] - p2[0] * p2[0] + p3[1] * p3[1] - p2[1] * p2[1]
        y0 = (x32 * xy21 - x21 * xy32) / (2 * (y21 * x32 - y32 * x21))
        x0 = (xy21 - 2 * y0 * y21) / (2.0 * x21)
        R = ((p1[0] - x0) ** 2 + (p1[1] - y0) ** 2) ** 0.5
        return x0, y0, R


    def is_collision(self, agent1, agent2, safe_dis = 0.0):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # TODO dist_min may need modification due to real robots in gazebo
        dist_min = agent1.size + agent2.size + safe_dis / 25.0
        return dist < dist_min


    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        boundary_reward = -100 if self.outside_boundary(agent) else 0
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        # Manully tuned for building areas
        #print(entity_a.name, entity_b.name)
        obs_avoid_reward = -100 if (agent.ugv and self.in_range(agent, [0,1,2,3])) or (not agent.ugv and self.in_range(agent, [2])) else 0
        return main_reward + boundary_reward + obs_avoid_reward

    # The boundary of screen
    def outside_boundary(self, agent):
        return agent.state.p_pos[0] > 1 or agent.state.p_pos[0] < -1 or agent.state.p_pos[1] > 1 or agent.state.p_pos[1] < -10.0/41.25

    # is it in a building where it shouldn't be?
    def in_range(self, agent, range_id=[0,1,2,3]):
        range = [[[-25.0,-14.0],[-10.0,29.0]],
                    [[-14.0,0.0],[15.0,29.0]],
                    [[-4.0,4.0],[-6,0,6.0]],
                    [[4.0,15.0],[-6.0,18.0]]] # [[x_min,x_max],[y_min,y_max]]
        x = agent.state.p_pos[0] * 25.0
        y = agent.state.p_pos[1] * 41.25
        for i in range_id:
            if x > range[i][0][0] and x < range[i][0][1] and \
                    y > range[i][1][0] and y < range[i][1][1]:
                return True
        return False


    def agent_reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        if shape:
            for adv in adversaries:
                #old parameter is 0.5 * ...
                rew += 1.2 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos))) # The further, the better
        if agent.collide:
            i = 0
            for a in adversaries:
                # Penalty for being caught
                if self.is_collision(a, agent, 3.5):
                    i += 1

            # too strict
            if i == self.num_cooperator:
                rew -= 50
                #print("Caught(good) :-(")
            else:
                rew -= 1 * abs(self.num_cooperator - i)

            # simplified
            #rew -= 3 * i

        '''
        assert world.dim_p == 2
        x = agent.state.p_pos[0]
        if x > 1.0 or x < -1.0:
            rew -= np.exp(10 * (abs(x) - 1.0))
        elif x > 0.9 or x < -0.9:
            rew -= 10 * (abs(x) - 1.0)
        y = agent.state.p_pos[1]
        if y > 1.0:
            rew -= np.exp(10 * (abs(y) - 1.0))
        elif y < -10.0 / 41.25:
            rew -= np.exp(10 * (abs(y) - 10.0 / 41.25))
            #if not bound(x) == 0:
                #print("bound")
        '''
        # Is food good for training? Because actually with a static objective, agents will tremble
        '''
        for food in world.food:
            if self.is_collision(agent, food):
                rew += 1
                #print("food")
        '''
        # I think this is better for food
        #rew += 0.1 * min([np.sqrt(np.sum(np.square(food.state.p_pos - agent.state.p_pos))) for food in world.food])
        rew -= 0.1 * min([np.sqrt(np.sum(np.square(food.state.p_pos - agent.state.p_pos))) for food in world.food]) # want to get nearer to one of the foods
        return rew

    def adversary_reward(self, agent, world):
        # Adversary agents are rewarded based on minimum agent distance to each good agent(Shape:true)
        rew = 0
        succ = 0
        shape = True
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:
            rew -= 1.0 * min([np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) for a in agents]) # The further the worse
        if agent.collide:
            for ag in agents:
                i = 0
                for adv in adversaries:
                    if self.is_collision(ag, adv, 3.5):
                        i += 1

                #if i: #print(i)
                # too strict
                if i == self.num_cooperator:
                    rew += 50
                    succ += 1
                else:
                    rew += 1 * abs(self.num_cooperator - i)

                # simplified
                #rew += 5 * i

        # Calculate success rate(precision)
        '''
        if "0" in agent.name and succ:
            fr = open("/home/crai/results/metrics/succ_rate_maddpg.txt","r")
            old_succ = fr.readline()
            fr.close()
            fw = open("/home/crai/results/metrics/succ_rate_maddpg.txt","w")
            fw.write(str(int(old_succ.strip('\n'))+succ))
            fw.close()
        '''
        return rew

    # Observation settings below is very important!! Keep in mind!! Need more modifications!!

    # This is not used
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


    # Called in _get_obs(self, agent) of environment.py
    # get observation for a particular agent
    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                # Do I need to add observation range for buildings? Actually, it has to be the same with observation of other agents to make sense!
                if "lawn" in entity.name:
                    # Polygon center
                    landmark_id = entity.name[-1]
                    coor = ksu.lawn_coordinations[int(landmark_id)]
                    if len(coor) == 4:
                        center_x, center_y = 0.0, 0.0
                        for i in range(4):
                            center_x += coor[i][0]
                            center_y += coor[i][1]
                        p_pos = np.array([center_x / 4.0 / 25.0, center_y / 4.0 / 41.25])
                    elif len(coor) == 3:
                    # Triangle center
                        center_x, center_y, _ = self.getCircle(coor[0], coor[1], coor[2])
                        p_pos = np.array([center_x / 25.0, center_y / 41.25])
                    entity_pos.append(p_pos - agent.state.p_pos)
                else:
                    entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # Self in forest detection, only for ugv
        in_forest = [np.array([-1]), np.array([-1])]
        inf1 = False
        inf2 = False
        if agent.ugv and self.is_collision(agent, world.forests[0]):
            in_forest[0] = np.array([1])
            inf1= True
        if agent.ugv and self.is_collision(agent, world.forests[1]):
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
            #if (inf1 and oth_f1) or (inf2 and oth_f2) or (not inf1 and not oth_f1 and not inf2 and not oth_f2):# or agent.leader:
            if (agent.ugv and self.is_collision(agent, other, safe_dis=20.0)) or not (agent.ugv or oth_f1 or oth_f2):
                other_pos.append(np.array(other.state.p_pos) - agent.state.p_pos)
                if not other.adversary:
                    other_vel.append(other.state.p_vel)
            else:
                other_pos.append([0, 0])
                if not other.adversary:
                    other_vel.append([0, 0])

            # for performance evaluation
            '''
            if not other.adversary and agent.adversary:
                delta_pos = other.state.p_pos - agent.state.p_pos
                #dist = np.sqrt(np.sum(np.square(delta_pos)))
                dist = np.sqrt(np.square(delta_pos[0])+np.square(delta_pos[1] / 1.65))
                fw = open("/home/crai/results/metrics/dists.log","a+")
                fw.write(str(dist)+",")
                fw.close()
            '''

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

        if agent.adversary:# and not agent.leader:
            return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel + in_forest + comm)
        #if agent.leader: # and is agent.adversary
        #    return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel + in_forest + comm)
        else:
            # only good agents themselves know their sweet points(food); their are not communicated with adveraries
            return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel + in_forest + food_pos) # change for good agent who will chase food at the same time
