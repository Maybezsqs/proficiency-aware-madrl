# Proficiency Aware Multi-Agent Actor-Critic for Mixed Robot Teaming (Mix-DRL)

## Multi-Agent Particle Environment

A simple multi-agent particle world with a continuous observation and discrete action space, along with some basic simulated physics. From OpenAI [Multi-Agent Particle Environments (MPE)](https://github.com/openai/multiagent-particle-envs).

## Getting started:

- To install, `cd` into the root directory and type `pip install -e .`

- Ensure that multiagent-particle-envs has been added to your PYTHONPATH (e.g. in ~/.bashrc or ~/.bash_profile).

- Install `mixdrl` by `cd` into the `mixdrl` directory and type `pip install -e .`

- Known dependencies: Python (3.5.4), OpenAI gym (0.10.5), numpy (1.14.5), tensorflow (1.8.0). Please see `environment.yml` for the complete dependencies.

- To run the code, `cd` into the `mixdrl/experiments` directory and run `train.py`: `python train.py --scenario simple_world_comm`. You can replace this experiment scenario with `simple_world_comm_5vs2`.

## Code structure

- `make_env.py`: contains code for importing a multiagent environment as an OpenAI Gym-like object.

- `./multiagent/environment.py`: contains code for environment simulation (interaction physics, `_step()` function, etc.)

- `./multiagent/core.py`: contains classes for various objects (Entities, Landmarks, Agents, etc.) that are used throughout the code.

- `./multiagent/rendering.py`: used for displaying agent behaviors on the screen.

- `./multiagent/policy.py`: contains code for interactive policy based on keyboard input.

- `./multiagent/scenario.py`: contains base scenario object that is extended for all scenarios.

- `./multiagent/scenarios/`: folder where various scenarios/ environments are stored. scenario code consists of several functions:
    1) `make_world()`: creates all of the entities that inhabit the world (landmarks, agents, etc.), assigns their capabilities (whether they can communicate, or move, or both).
     called once at the beginning of each training session
    2) `reset_world()`: resets the world by assigning properties (position, color, etc.) to all entities in the world
    called before every episode (including after make_world() before the first episode)
    3) `reward()`: defines the reward function for a given agent
    4) `observation()`: defines the observation space of a given agent
    5) (optional) `benchmark_data()`: provides diagnostic data for policies trained on the environment (e.g. evaluation metrics)


## List of environments

| Env name in code (name in paper) |  Communication? | Competitive? | Notes |
| --- | --- | --- | --- |
| `simple_world_comm.py` | Y | Y | Environment seen in the video accompanying the paper. Same as simple_tag, except (1) there is food (small blue balls) that the good agents are rewarded for being near, (2) we now have ‘forests’ that hide agents inside from being seen from outside; (3) there is a ‘leader adversary” that can see the agents at all times, and can communicate with the other adversaries to help coordinate the chase. |