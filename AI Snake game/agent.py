import math
import torch
import torch.nn.functional as F
import random
import json
import numpy as np
import os
from collections import deque
from game import SnakeGameAI, Point, BLOCK_SIZE, Direction  # Added Direction
from model import ActorCritic, PPOTrainer, RolloutBuffer

class Agent:
    def __init__(self):
        self.settings = {
            "batch_size": 64,
            "lr": 0.0001,  # Lower for deeper net stability
            "gamma": 0.99,
            "grid_size": 40,
            "channels": 4,  # Now 4: obs, food, head, tail
            "file_name": "model.pth",
            "render_ui": False,
            "epsilon_start": 0.9,
            "epsilon_min": 0.05,
            "epsilon_decay": 0.99985,  # Slightly slower for endgame expl
            "ppo_update_steps": 8192,  # Longer for fill horizons
            "ppo_epochs": 15,
            "ppo_clip_eps": 0.2,
            "ppo_gae_lambda": 0.95,
            "ppo_ent_coef": 0.01,  # Reduced as policy sharpens
            "ppo_vf_coef": 0.5,
            "ppo_batch_size": 64,
            "ppo_max_grad_norm": 0.5,
            "use_epsilon": True
        }   

        try:
            with open('settings.json', 'r') as file:
                loaded = json.load(file)
                self.settings.update(loaded)
                print("Loaded settings.json")
        except FileNotFoundError:
            print("settings.json not found — using defaults")
        except Exception as e:
            print(f"Error loading settings.json: {e} — using defaults")

        self.n_games = 0
        self.gamma = self.settings['gamma']
        self.ac_net = ActorCritic(self.settings['grid_size'], self.settings['channels'])
        self.game_history = deque()
        
        # Load model
        model_dir = 'model'
        model_path = os.path.join(model_dir, self.settings['file_name'])
        os.makedirs(model_dir, exist_ok=True)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'actor_critic' in checkpoint:
                self.ac_net.load_state_dict(checkpoint['actor_critic'])
                print(f"Loaded PPO model from {model_path}")
            else:
                print("Old format; starting fresh.")

        self.trainer = PPOTrainer(
            self.ac_net,
            lr=self.settings['lr'],
            gamma=self.gamma,
            clip_eps=self.settings['ppo_clip_eps'],
            epochs=self.settings['ppo_epochs'],
            gae_lambda=self.settings['ppo_gae_lambda'],
            ent_coef=self.settings['ppo_ent_coef'],
            vf_coef=self.settings['ppo_vf_coef'],
            batch_size=self.settings['ppo_batch_size'],
            max_grad_norm=self.settings['ppo_max_grad_norm']
        )

        self.rollout_buffer = RolloutBuffer()

        self.epsilon = self.settings['epsilon_start']
        self.epsilon_min = self.settings['epsilon_min']
        self.epsilon_decay = self.settings['epsilon_decay']
        self.use_epsilon = self.settings['use_epsilon']

        self.update_steps = self.settings['ppo_update_steps']

    def get_grid_state(self, game):
        grid_size = self.settings['grid_size']
        state = np.zeros((self.settings['channels'], grid_size, grid_size), dtype=np.float32)

        # Offsets
        actual_cols = game.w // BLOCK_SIZE
        actual_rows = game.h // BLOCK_SIZE
        offset_x = (grid_size - actual_cols) // 2
        offset_y = (grid_size - actual_rows) // 2

        # Channel 0: Obstacles (outside + walls + body[1:])
        state[0, :, :] = 1.0 
        state[0, offset_y:offset_y + actual_rows, offset_x:offset_x + actual_cols] = 0.0
        for wall in game.walls:
            gx = (wall.x // BLOCK_SIZE) + offset_x
            gy = (wall.y // BLOCK_SIZE) + offset_y
            if 0 <= gx < grid_size and 0 <= gy < grid_size:
                state[0, gy, gx] = 1.0
        for pt in game.snake[1:]:
            gx = (pt.x // BLOCK_SIZE) + offset_x
            gy = (pt.y // BLOCK_SIZE) + offset_y
            if 0 <= gx < grid_size and 0 <= gy < grid_size:
                state[0, gy, gx] = 1.0

        # Channel 1: Food (1.0), Special (2.0)
        if game.food:
            fx, fy = (game.food.x // BLOCK_SIZE) + offset_x, (game.food.y // BLOCK_SIZE) + offset_y
            if 0 <= fx < grid_size and 0 <= fy < grid_size:
                state[1, fy, fx] = 1.0
        if game.special_food:
            sx, sy = (game.special_food.x // BLOCK_SIZE) + offset_x, (game.special_food.y // BLOCK_SIZE) + offset_y
            if 0 <= sx < grid_size and 0 <= sy < grid_size:
                state[1, sy, sx] = 2.0

        # Channel 2: Head
        hx, hy = (game.head.x // BLOCK_SIZE) + offset_x, (game.head.y // BLOCK_SIZE) + offset_y
        if 0 <= hx < grid_size and 0 <= hy < grid_size:
            state[2, hy, hx] = 1.0

        # Channel 3: Tail
        if len(game.snake) > 0:
            tx, ty = (game.snake[-1].x // BLOCK_SIZE) + offset_x, (game.snake[-1].y // BLOCK_SIZE) + offset_y
            if 0 <= tx < grid_size and 0 <= ty < grid_size:
                state[3, ty, tx] = 1.0

                # Head-centric transformation
        center = grid_size // 2
        shift_x = center - hx
        shift_y = center - hy
        state_shifted = np.roll(np.roll(state, shift_y, axis=1), shift_x, axis=2)

        # Rotate to canonical (head faces RIGHT)
        rot_k = {
            Direction.RIGHT: 0,
            Direction.DOWN:  3,   # 270° CCW = 90° CW
            Direction.LEFT:  2,   # 180°
            Direction.UP:    1    # 90° CCW
        }[game.direction]

        state_rotated = np.rot90(state_shifted, k=rot_k, axes=(1, 2))

        # Critical: force contiguous memory layout → fixes negative stride issue
        state_canonical = state_rotated.copy(order='C')   # or just .copy()

        return state_canonical

    def get_state(self, game):
        return self.get_grid_state(game)

    def get_action(self, state, train=True):
        state_t = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        logits, value = self.ac_net(state_t)
        
        if train and self.use_epsilon and random.random() < self.epsilon:
            action_idx = random.randint(0, 2)
            logp_a = 0.0
        else:
            probs = F.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(probs)
            action_idx_tensor = dist.sample()
            action_idx = action_idx_tensor.item()
            logp_a = dist.log_prob(action_idx_tensor).item()
        
        action_onehot = F.one_hot(torch.tensor(action_idx), num_classes=3).squeeze(0).cpu().numpy().tolist()
        
        return action_onehot, action_idx, logp_a, value.squeeze().item()

def train():
    agent = Agent()
    game = SnakeGameAI(agent.settings['render_ui'])
    record = 0

    while True:
        state_old = agent.get_state(game)
        
        action_onehot, action_idx, logp, value = agent.get_action(state_old, train=True)
        reward, level, done, score, won, duration = game.play_step(action_onehot)
        state_new = agent.get_state(game)
        
        agent.rollout_buffer.add(state_old, action_idx, logp, value, reward, bool(done))
        
        if done or won:
            agent.trainer.update(agent.rollout_buffer, next_value=0.0)
            agent.rollout_buffer.clear()
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay
        elif agent.rollout_buffer.size() >= agent.update_steps:
            with torch.no_grad():
                _, next_value = agent.ac_net(torch.tensor(state_new, dtype=torch.float).unsqueeze(0))
            agent.trainer.update(agent.rollout_buffer, next_value.squeeze().item())
            agent.rollout_buffer.clear()
        
        if done or won:
            reason = game.death_reason if not won else "WIN 🎉"
            game.reset()
            agent.n_games += 1

            if score > record:
                record = score
                agent.ac_net.save(agent.settings['file_name'])

            msg = f'Game {agent.n_games} | Score: {score} | Record: {record} | Level: {level} | Epsilon: {agent.epsilon:.3f} | Reason: {reason} | Fill: {game.get_fill_ratio():.2f}%'
            if won:
                msg += f" | STEPS: {game.frame_iteration} | TIME: {duration}"
                print(msg)
            else:
                print(msg)

            agent.game_history.append({'score': score, 'level': level, 'won': won, 'fill': game.get_fill_ratio()})
            if len(agent.game_history) % 100 == 0:
                avg_score = np.mean([g['score'] for g in agent.game_history])
                avg_level = np.mean([g['level'] for g in agent.game_history])
                print(f"--- 100-GAME AVG: Score {avg_score:.1f} | Level {avg_level:.1f} | Epsilon {agent.epsilon:.3f} ---")
                agent.game_history.clear()

if __name__ == '__main__':
    train()