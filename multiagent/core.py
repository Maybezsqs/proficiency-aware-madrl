import numpy as np

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None
        # TODO no yaw in python simulation

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        # Agents controlling: Either by policies or scripts
        # Default: None
        self.action_callback = None


# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        # landmarks=buildings,food,forests,lawns
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2 # x and y
        # color dimensionality
        self.dim_color = 3 # rgb
        # simulation timestep TODO
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

        # for KSU map coordinations
        self.building_coordinations = []
        self.lawn_coordinations = []
        self.forest_coordinations = []
        self.food_coordinations = []

    # return all entities in the world
    @property
    def entities(self):
        # the order determines the layer order(upfront or back)
        return self.landmarks + self.agents # first landmarks then agents

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i,agent in enumerate(self.entities):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.action.u + noise
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a,entity_a in enumerate(self.entities):
            for b,entity_b in enumerate(self.entities):
                if(b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
        return p_force

    # integrate physical state
    def integrate_state(self, p_force): # It is this function that actually changes the state(the position)
        j = 0
        
        #fw = open("/home/crai/results/trajectory/trajectory.txt","a")
        
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            j += 1
            # Only for agents
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1])) # 1.65?
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / speed * entity.max_speed
            entity.state.p_vel[1] /= 1.65 # TODO scaling set_bounds in environment.py
            entity.state.p_pos += entity.state.p_vel * self.dt

            # output trajectory
            '''
            fw.write(str(entity.state.p_pos[0]*25.0)+","+str(entity.state.p_pos[1]*41.25))
            if j < len(self.agents):
                fw.write(",")
            else:
                fw.write("\n")
                fw.close()
            '''

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent: # Then say nothing
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise

    # get collision forces for any contact between two entities (a < b)
    # TODO different perception of the world with different robots
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        if (not entity_a.movable and not entity_b.movable):
            return [None, None] # Non-movable objects don't collide with each other
        if ("6" not in entity_a.name and not entity_b.ugv):
            return [None, None] # uav has more reachable spaces than ugv except for the tallest library building
        '''
        if 'agent' in entity_a.name:
            # compute actual distance between two agents(entities)
            delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            # minimum allowable distance
            dist_min = entity_a.size + entity_b.size # + 2.0 / 25.0
        elif 'agent' in entity_b.name:
            # a is landscape but b is agent
            import multiagent.scenarios.ksu_map as ksu
            import math
            if 'landmark' in entity_a.name:
                # Consider agent collide with landmark
                # Exclude uav with small buildings
                # TODO change the number of UAV and UGV
                #if ("0" or "1" in entity_b.name) and ("4" not in entity_a.name):
                #    return [None, None] # UAV can fly above some buildings except for building 4
                landmark_id = entity_a.name[-1]
                coor = ksu.building_coordinations[int(landmark_id)]
                center_x, center_y = 0.0, 0.0
                for i in range(len(coor)):
                    center_x += coor[i][0]
                    center_y += coor[i][1]
                p_pos = (center_x / 4.0, center_y / 4.0)
                # compute actual distance between the agent and the entity
                delta_pos = p_pos - entity_b.state.p_pos
                dist = np.sqrt(np.sum(np.square(delta_pos)))
                # minimum allowable distance (safest!!!)
                building_range = max(np.sqrt(np.sum(np.square(np.array(p_pos)- c))) for c in coor) / 41.25 # np.sqrt(np.sum(np.square(np.array([25.0,41.25]))))
                dist_min = building_range + entity_b.size + 1.0 / 25.0
        else:
            # Two building landmarks
            return [None, None]
        '''
        # compute actual distance between two agents(entities)
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size + 1.0 / 25.0
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]


