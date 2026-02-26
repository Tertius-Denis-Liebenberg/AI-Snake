# 🐍 Snake AI – PPO Reinforcement Learning

*A deep reinforcement learning agent that learns to play an advanced Snake game using Proximal Policy Optimization (PPO).  
Trained to survive longer, fill the board efficiently, avoid traps, and complete increasingly difficult levels with dynamic walls and special food.*

---

<p align="center">
  <img src="https://img.shields.io/badge/Algorithm-PPO-FF6F61?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/State-Grid%20CNN%20%2B%20ActorCritic-blue" />
  <img src="https://img.shields.io/badge/Environment-Custom%20Snake%20Game-green" />
  <img src="https://img.shields.io/badge/Status-In%20Development-yellow" />
  <img src="https://img.shields.io/badge/PRs-Welcome-brightgreen" />
</p>

---

## 📖 Project Description

This project trains a **PPO-based reinforcement learning agent** to master an enhanced version of the classic Snake game.

### Game Features
- 5 progressive difficulty levels with growing grid size and more walls
- Special food (bonus points + temporary high reward)
- Fill-based progression (agent rewarded for high board coverage %)
- Trap detection & penalties (dead-end spaces)
- Dynamic danger signals, looping penalties, timeout pressure
- Visual feedback: glowing snake, pulsing special food, HUD with fill %

### Learning Features
- Grid observation with 4 channels (obstacles, food/special, head, tail)
- Head-centric + rotation-invariant state (canonical orientation)
- PPO with GAE, entropy bonus, clipped surrogate, value function
- Epsilon-greedy exploration (decaying)
- Frequent updates + long rollouts tuned for long-horizon filling task
- Checkpoint saving on new high scores

Goal: teach the agent not just to eat food, but to **efficiently fill the board** while avoiding traps and walls — especially in later levels.

---

## ✨ Current Capabilities

- Reliable PPO training loop with rollout buffer & GAE advantages
- Multi-channel CNN input (40×40 grid)
- Custom reward shaping:
  - Strong food & special-food rewards
  - Fill percentage bonuses & milestones
  - Proximity to food (positive/negative delta)
  - Penalties: trap, looping, danger, wall hit, timeout
- Level progression & win condition (fill board completely)
- Visual rendering (optional) with glow effects & HUD
- Save/load model + high-score tracking

---

## 🛠️ Tech Stack

- **Language**: Python 3.8+
- **Deep Learning**: PyTorch
- **Game Engine**: Pygame (rendering + input)
- **Reinforcement Learning**: Custom PPO + Actor-Critic
- **Observation**: 4-channel 40×40 grid (CNN)
- **Hyperparameters**: Loaded from `settings.json` (easy tuning)
- **Rewards**: Configurable via `rewards.json`

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install torch torchvision torchaudio pygame numpy
```

### 2. Prepare files (optional but recommended)

Create or edit:
- `settings.json` → hyperparameters (lr, batch size, epsilon decay, PPO settings…)
- `rewards.json` → reward/penalty values

### 3. Run training

```bash
python agent.py
```

Training will:
- Load model if exists (`model/model.pth`)
- Print game stats every episode
- Show 100-game rolling averages
- Save best model on new record score

To watch the agent (slower but visual):

```bash
# In settings.json
"render_ui": true
```

---

## 🎮 Game Controls (human mode – not used in training)

- Arrow keys or WASD
- But training uses direct action vector [straight, right, left]

---

## 📊 Reward Structure (rewards.json)

```json
{
    "eat_food": 200,
    "eat_special_food": 150,
    "level_up": 150,
    "win": 500,
    "game_over": -50,
    "looping_penalty": -5,
    "hit_wall_penalty": -5,
    "into_danger_penalty": -20,
    "empty_space_bonus": 0.5,
    "trap_penalty": -40,
    "closer_to_food": 1.0,
    "away_from_food": -0.6,
    "survival_bonus": 0.1,
    "fill_milestone_bonus": 100
}
```

Heavily shaped to encourage **board filling** over just eating.

---

## 🗺️ Roadmap / Planned Improvements

- Curriculum learning (start easier → gradually harder levels)
- Self-play or opponent snakes
- Larger grid / variable sizes
- Frame stacking for velocity awareness
- Attention or transformer backbone
- Video recording of best episodes
- TensorBoard / Weights & Biases logging
- Multi-agent or competitive mode

---

## 🌟 Why This Project?

Classic Snake + RL is a fantastic testbed for long-horizon credit assignment, sparse rewards, and exploration.
This version pushes the challenge further with:

- Fill-the-board objective
- Trap avoidance
- Progressive difficulty
- Rich reward shaping

It's both fun to watch and a serious test of PPO stability on a partially observable, long-episode task.

---

## 🤝 Contact

* Maintainer: Tertius Denis Liebenberg
* Email: [tertiusliebenberg7@gmail.com](mailto:tertiusliebenberg7@gmail.com)
* GitHub: [Tertius Denis Liebenberg](https://github.com/Tertius-Denis-Liebenberg)
