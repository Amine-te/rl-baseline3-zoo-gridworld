"""
Training script for GridWorld environments with rl-baselines3-zoo.
This script imports gridworld to register custom environments before training.

Usage:
    python train_gridworld.py --algo dqn --env GridWorld-v0 -n 10000
    python train_gridworld.py --algo ppo --env GridWorld-v1 -n 20000
    python train_gridworld.py --algo dqn --env GridWorld-v1 -n 10000 -f logs/
"""

# Import gridworld to register the environment
import gridworld

# Now run the normal training
from rl_zoo3.train import train

if __name__ == "__main__":
    train()