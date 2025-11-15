# RL-Baselines3-Zoo + Custom GridWorld Environment

This repository is based on the original **rl-baselines3-zoo** project.  
I extended it by integrating a **custom GridWorld environment** and training several agents on it using Stable-Baselines3.

---

## 🔧 Overview of Added Features

### **1. Custom GridWorld Environment**
- Implemented a custom `GridWorld-v0` Gymnasium environment.
- Compatible with Stable-Baselines3 and the zoo training pipeline.
- Includes reward shaping, obstacles, terminal states, etc.

### **2. Training Agents on GridWorld**
Trained the following RL algorithms on the environment:
- **DQN**
- **PPO**
- **A2C**
- (Add/remove as appropriate)

Training was done through the RL-Baselines3-Zoo framework:
```bash
python train.py --algo dqn --env GridWorld-v0 --tensorboard logs/
