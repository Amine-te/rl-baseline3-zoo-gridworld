from gymnasium.envs.registration import register
from gridworld.gridworld import GridWorldEnv

# Basic 5x5 GridWorld
register(
    id='GridWorld-v0',
    entry_point='gridworld.gridworld:GridWorldEnv',
    kwargs={
        'grid_size': 5,
        'goals': [[4, 4]],
        'start_pos': [0, 0],
        'obstacles': [],
        'reward_goal': 10.0,
        'reward_step': -0.1,
        'max_steps': 50,
        'state_type': 'absolute'
    }
)

# 10x10 GridWorld with obstacles
register(
    id='GridWorld-v1',
    entry_point='gridworld.gridworld:GridWorldEnv',
    kwargs={
        'grid_size': 10,
        'goals': [[9, 9]],
        'start_pos': [0, 0],
        'obstacles': [
            [2, 2], [2, 3], [2, 4], [2, 5],
            [5, 1], [5, 2], [5, 3],
            [7, 5], [7, 6], [7, 7], [7, 8],
            [4, 7], [5, 7], [6, 7]
        ],
        'reward_goal': 10.0,
        'reward_step': -0.1,
        'max_steps': 100,
        'state_type': 'absolute'
    }
)