import gymnasium as gym
import gridworld
import imageio
import numpy as np
import os
from stable_baselines3 import DQN, PPO
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def record_agent(env_id, model, filename, max_steps=100):
    """Record an agent's trajectory as a GIF in a GridWorld environment."""
    env = gym.make(env_id)
    obs, info = env.reset()
    frames = []

    for step in range(max_steps):
        # Create figure
        fig, ax = plt.subplots(figsize=(6, 6))
        grid_size = env.unwrapped.height
        width = env.unwrapped.width

        # Draw grid
        for i in range(grid_size + 1):
            ax.axhline(i, color='black', linewidth=1)
        for i in range(width + 1):
            ax.axvline(i, color='black', linewidth=1)

        # Draw goals
        for goal in env.unwrapped.goals:
            gx, gy = goal[1], grid_size - goal[0] - 1
            ax.add_patch(Rectangle((gx, gy), 1, 1, facecolor='red', edgecolor='darkred', linewidth=3))

        # Draw obstacles
        for obs_pos in env.unwrapped.obstacles:
            ox, oy = obs_pos[1], grid_size - obs_pos[0] - 1
            ax.add_patch(Rectangle((ox, oy), 1, 1, facecolor='gray', edgecolor='black', linewidth=2))

        # Draw agent
        ax_x, ax_y = env.unwrapped.agent_pos[1], grid_size - env.unwrapped.agent_pos[0] - 1
        ax.add_patch(Rectangle((ax_x, ax_y), 1, 1, facecolor='green', edgecolor='darkgreen', linewidth=3))

        ax.set_xlim(0, width)
        ax.set_ylim(0, grid_size)
        ax.set_aspect('equal')
        ax.set_title(f'{env_id} - Step {step}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Convert to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        frames.append(image)
        plt.close(fig)

        # Model action
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            for _ in range(10):  # Hold final frame for ~1 sec
                frames.append(image)
            break

    # Save as GIF
    os.makedirs("gif", exist_ok=True)
    path = os.path.join("gif", filename)
    imageio.mimsave(path, frames, fps=10)
    print(f"✅ GIF saved as '{path}'!")

    env.close()


# ---- MAIN EXECUTION ----

# GridWorld-v0 (5x5)
print("\n=== Recording GridWorld-v0 (5x5) ===")
env_id_v0 = 'GridWorld-v0'

# DQN v0
dqn_model_v0 = DQN.load("logs/dqn/GridWorld-v0_1/GridWorld-v0.zip")
record_agent(env_id_v0, dqn_model_v0, "gridworld_v0_dqn.gif")

# PPO v0
ppo_model_v0 = PPO.load("logs/ppo/GridWorld-v0_1/GridWorld-v0.zip")
record_agent(env_id_v0, ppo_model_v0, "gridworld_v0_ppo.gif")


# GridWorld-v1 (10x10 with obstacles)
print("\n=== Recording GridWorld-v1 (10x10 with obstacles) ===")
env_id_v1 = 'GridWorld-v1'

# DQN v1
dqn_model_v1 = DQN.load("logs/dqn/GridWorld-v1_1/GridWorld-v1.zip")
record_agent(env_id_v1, dqn_model_v1, "gridworld_v1_dqn.gif", max_steps=150)

# PPO v1
ppo_model_v1 = PPO.load("logs/ppo/GridWorld-v1_1/GridWorld-v1.zip")
record_agent(env_id_v1, ppo_model_v1, "gridworld_v1_ppo.gif", max_steps=150)

print("\n✨ All GIFs created successfully!")