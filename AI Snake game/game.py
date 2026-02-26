import os
import pygame
import random
from enum import Enum
from collections import namedtuple, deque
import math
import time
import numpy as np
import json

pygame.init()

# ----- Constants -----
BLOCK_SIZE = 20
SPEED_BASE = 40

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (200, 0, 0)
YELLOW = (255, 255, 0)
DARK_GRID = (15, 15, 25)
WALL_COLOR = (80, 80, 100)
WALL_GLOW = (40, 40, 60)

SNAKE_COLORS = [
    (0, 180, 255),   # Level 1
    (0, 255, 120),   # Level 2
    (160, 80, 255),  # Level 3
    (255, 140, 60),  # Level 4
    (255, 215, 0),   # Level 5
]

font = pygame.font.Font('Default.ttf', 20)

# ----- Sounds -----
def load_sound(path):
    try:
        return pygame.mixer.Sound(path)
    except:
        return None

EAT_SOUND = load_sound("Sound Effects/eating.mp3")
SPECIAL_FOOD_SOUND = load_sound("Sound Effects/special_food.mp3")
GAME_OVER_SOUND = load_sound("Sound Effects/game_over.mp3")
LEVEL_UP_SOUND = load_sound("Sound Effects/level_up.mp3")
GAME_COMPLETE = load_sound("Sound Effects/win.mp3")

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

class SnakeGameAI:
    def __init__(self, render=True):
        self.render = render
        self.clock = pygame.time.Clock()
        self.high_score = self._load_high_score()

        # Fast lookup sets
        self.wall_set = set()

        self.reset()

    def reset(self, level=1):
        try:
            with open('rewards.json', 'r') as file:
                self.default_rewards = json.load(file)
        except:
            print('No rewards file found! Using defaults.')
            self.default_rewards = {
                "eat_food": 25,
                "eat_special_food": 75,
                "level_up": 60,
                "win": 150,
                "game_over": -50,
                "looping_penalty": -2,
                "hit_wall_penalty": -4,
                "into_danger_penalty": -10,
                "empty_space_bonus": 0.1,
                "trap_penalty": -20,
                "closer_to_food": 0.4,
                "away_from_food": -0.25
            }

        self.start_time = time.time()
        self.current_level = level
        self.max_levels = 5
        self.score = 0
        self.frame_iteration = 0
        self.recent_positions = deque(maxlen=20)
        self.death_reason = ""
        self.game_won = False
        self._init_level_properties(self.current_level)

        # Track fill milestones for rewards (25%, 50%, 75%)
        self.fill_milestones = [0.25, 0.5, 0.75]
        self.achieved_milestones = []

    def _init_level_properties(self, level):
        base_w, base_h = 400, 400
        extra = (level - 1) * 100
        self.w = base_w + extra
        self.h = base_h + extra

        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption(f'Snake Game - Level {level}')

        start_x = (self.w // 2) // BLOCK_SIZE * BLOCK_SIZE
        start_y = (self.h // 2) // BLOCK_SIZE * BLOCK_SIZE
        self.head = Point(start_x, start_y)
        self.direction = Direction.RIGHT

        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2 * BLOCK_SIZE, self.head.y)
        ]

        self.walls = []
        self._generate_walls(level)
        self.wall_set = set(self.walls)  # exclude head

        total_slots = (self.w // BLOCK_SIZE) * (self.h // BLOCK_SIZE)
        self.max_capacity = total_slots - len(self.walls)

        self.food = None
        self.special_food = None
        self.special_food_timer = 0
        self._place_food()

    def _generate_walls(self, level):
        cols = self.w // BLOCK_SIZE
        rows = self.h // BLOCK_SIZE
        self.walls = []

        if level == 1:
            pass
        elif level == 2:
            for x in [5, cols-6]:
                for y in [5, rows-6]:
                    self.walls.append(Point(x*BLOCK_SIZE, y*BLOCK_SIZE))
        elif level == 3:
            y_positions = [rows//3, (rows//3)*2]
            for y in y_positions:
                for x in range(4, cols-4):
                    self.walls.append(Point(x*BLOCK_SIZE, y*BLOCK_SIZE))
        elif level == 4:
            center_x = cols // 2
            center_y = rows // 2
            for x in range(center_x - 5, center_x + 5):
                for y in range(center_y - 5, center_y + 5):
                    if x in (center_x-5, center_x+4) or y in (center_y-5, center_y+4):
                        self.walls.append(Point(x*BLOCK_SIZE, y*BLOCK_SIZE))
        elif level == 5:
            random.seed(42)
            for x in range(0, cols, 2):
                for y in range(0, rows, 2):
                    if random.random() > 0.7:
                        if abs(x - cols//2) > 4 or abs(y - rows//2) > 4:
                            self.walls.append(Point(x*BLOCK_SIZE, y*BLOCK_SIZE))
                            self.walls.append(Point((x+1)*BLOCK_SIZE, y*BLOCK_SIZE))

    def _place_food(self):
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            candidate = Point(x, y)
            if (candidate not in self.snake and
                candidate != self.special_food and
                candidate not in self.wall_set):
                self.food = candidate
                break

    def _place_special_food(self):
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            candidate = Point(x, y)
            if (candidate not in self.snake and
                candidate != self.food and
                candidate not in self.wall_set):
                self.special_food = candidate
                self.special_food_timer = 0
                if SPECIAL_FOOD_SOUND: SPECIAL_FOOD_SOUND.play()
                break

    def _load_high_score(self):
        if os.path.exists("highscore.txt"):
            try:
                with open("highscore.txt", "r") as f:
                    return int(f.read())
            except:
                return 0
        return 0

    def _save_high_score(self):
        with open("highscore.txt", "w") as f:
            f.write(str(self.high_score))

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        reward = 0
        game_over = False
        game_won = False

        # Pre-move distances
        food_dist = math.hypot(self.head.x - self.food.x, self.head.y - self.food.y)
        special_dist = math.hypot(self.head.x - self.special_food.x, self.head.y - self.special_food.y) if self.special_food else float('inf')

        if self.game_won:
            reward = self.default_rewards['win']
            game_over = True
            game_won = True
            return reward, self.current_level, game_over, self.score, game_won, self._get_elapsed_time()

        self._move(action)
        self.snake.insert(0, self.head)

        is_looping = self.head in self.recent_positions
        self.recent_positions.append(self.head)

        if self.is_collision():
            if GAME_OVER_SOUND: GAME_OVER_SOUND.play()
            game_over = True
            reward = self.default_rewards['game_over']
            if self.death_reason == "Border" or self.death_reason == "Wall":
                reward += self.default_rewards['hit_wall_penalty']
            if self.score > self.high_score:
                self.high_score = self.score
                self._save_high_score()
            return reward, self.current_level, game_over, self.score, game_won, self._get_elapsed_time()

        # Trap check (additive now)
        if self.is_trap(self.head):
            reward += self.default_rewards['trap_penalty']

        # Timeout
        timeout_limit = 4000 + 800 * len(self.snake)
        fill_ratio = self.get_fill_ratio()
        if fill_ratio > 70:
            timeout_limit *= 1.5
        if fill_ratio > 90:
            timeout_limit *= 2.0  # Extra grace near end
        if self.frame_iteration > timeout_limit:
            self.death_reason = "Timeout"
            if GAME_OVER_SOUND: GAME_OVER_SOUND.play()
            game_over = True
            reward = self.default_rewards['game_over']
            if self.score > self.high_score:
                self.high_score = self.score
                self._save_high_score()
            return reward, self.current_level, game_over, self.score, game_won, self._get_elapsed_time()

        eaten = False
        if self.head == self.food:
            self.score += 1
            reward += self.default_rewards['eat_food']
            eaten = True
            self.frame_iteration = 0
            if len(self.snake) < self.max_capacity:
                self._place_food()
        elif self.head == self.special_food:
            self.score += 3
            reward += self.default_rewards['eat_special_food']
            eaten = True
            self.frame_iteration = 0
            self.special_food = None
        else:
            new_food_dist = math.hypot(self.head.x - self.food.x, self.head.y - self.food.y)
            delta_food = food_dist - new_food_dist
            reward += delta_food * self.default_rewards['closer_to_food'] * 1.5

            if delta_food < 0:
                reward += self.default_rewards['away_from_food'] * abs(delta_food)

            if self.special_food:
                new_special_dist = math.hypot(self.head.x - self.special_food.x, self.head.y - self.special_food.y)
                delta_special = special_dist - new_special_dist
                reward += delta_special * 1.2
                special_priority = max(0, delta_special * 1.5 - delta_food * 0.1)
                reward += special_priority

            if is_looping:
                reward += self.default_rewards['looping_penalty']

            immediate_danger = self._get_danger(self.head, self.direction)
            if immediate_danger > 0.5:
                reward -= self.default_rewards['into_danger_penalty']

            num_high_danger = self._get_multi_danger()
            if num_high_danger > 4:
                reward += self.default_rewards['trap_penalty']  # Redundant but reinforces

            self.snake.pop()

        # Always-applied bonuses/penalties (moved here for every step)
        fill_ratio = self.get_fill_ratio()  # Recalc after possible pop/eat
        reward += self.default_rewards['empty_space_bonus'] * (1 - fill_ratio / 100)
        reward -= 0.015  # Global time pressure for speed

        # Milestones (level-scaled)
        current_milestone = next((m for m in self.fill_milestones if fill_ratio >= m * 100 and m not in self.achieved_milestones), None)
        if current_milestone:
            milestone_reward = self.default_rewards.get('fill_milestone_bonus', 50) * current_milestone * self.current_level
            reward += milestone_reward
            self.achieved_milestones.append(current_milestone)

        # Special food spawn
        if self.score > 0 and self.score % 5 == 0 and self.special_food is None and not eaten:
            if len(self.snake) < self.max_capacity - 5:
                self._place_special_food()

        if self.special_food:
            self.special_food_timer += 1
            if self.special_food_timer > 100:
                self.special_food = None

        # Level completion
        if len(self.snake) >= self.max_capacity:
            if self.current_level < self.max_levels:
                self.current_level += 1
                reward += self.default_rewards['level_up'] * self.current_level
                if LEVEL_UP_SOUND: LEVEL_UP_SOUND.play()
                self._init_level_properties(self.current_level)
                self.achieved_milestones = []
                time.sleep(1)
            else:
                self.game_won = True
                if GAME_COMPLETE: GAME_COMPLETE.play()

        # Speed & UI
        speed = min(SPEED_BASE + (self.current_level - 1) * 2 + len(self.snake) * 0.1, 25)
        self.clock.tick(speed)
        self._update_ui()
        return reward, self.current_level, game_over, self.score, game_won, self._get_elapsed_time()

    def _get_multi_danger(self):
        ray_dirs = [(0,-1),(1,0),(0,1),(-1,0),(1,-1),(1,1),(-1,1),(-1,-1),(0.5,-1),(1,-0.5),(-0.5,-1),(-1,-0.5)]
        count = 0
        for dx, dy in ray_dirs:
            norm = math.sqrt(dx**2 + dy**2) or 1
            dx, dy = dx / norm, dy / norm
            danger = self._get_danger(self.head, Direction.RIGHT if dx > 0 else Direction.LEFT if dx < 0 else Direction.UP if dy < 0 else Direction.DOWN)
            if danger > 0.5:
                count += 1
        return count

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]
        else:
            new_dir = clock_wise[(idx - 1) % 4]
        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT: x += BLOCK_SIZE
        elif self.direction == Direction.LEFT: x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN: y += BLOCK_SIZE
        elif self.direction == Direction.UP: y -= BLOCK_SIZE
        self.head = Point(x, y)

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x >= self.w or pt.x < 0 or pt.y >= self.h or pt.y < 0:
            self.death_reason = "Border"
            return True
        if pt in self.snake[1:]:
            self.death_reason = "Tail"
            return True
        if pt in self.wall_set:
            self.death_reason = "Wall"
            return True
        return False
    
    def is_trap(self, point):
        # A fast Breadth-First Search (Flood Fill) to count accessible nodes
        # If accessible nodes < snake length, it's a trap.
        
        # Optimization: Don't run this every frame if board is huge, 
        # or only run if danger is detected.
        
        start_node = (point.x, point.y)
        if self.is_collision(point):
            return True

        visited = set()
        queue = deque([start_node])
        visited.add(start_node)
        count = 0
        limit = len(self.snake) # We only need to know if space >= snake length
        
        # Add current snake body to obstacles for this check
        obstacles = self.wall_set.union({(p.x, p.y) for p in self.snake})

        while queue:
            cx, cy = queue.popleft()
            count += 1
            if count > limit:
                return False # Safe, plenty of space

            for dx, dy in [(BLOCK_SIZE,0), (-BLOCK_SIZE,0), (0,BLOCK_SIZE), (0,-BLOCK_SIZE)]:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) not in visited and \
                   0 <= nx < self.w and 0 <= ny < self.h and \
                   (nx, ny) not in obstacles:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        
        return True # Trapped!

    def _get_danger(self, pt, direction):
        dx = dy = 0
        if direction == Direction.RIGHT: dx = BLOCK_SIZE
        elif direction == Direction.LEFT: dx = -BLOCK_SIZE
        elif direction == Direction.DOWN: dy = BLOCK_SIZE
        elif direction == Direction.UP: dy = -BLOCK_SIZE

        check = Point(pt.x + dx, pt.y + dy)

        if (check.x < 0 or check.x >= self.w or 
            check.y < 0 or check.y >= self.h or
            check in self.snake[1:] or 
            check in self.wall_set):
            return 1.0
        return 0.0

    def _get_elapsed_time(self):
        elapsed = int(time.time() - self.start_time)
        return f"{elapsed // 60:02}:{elapsed % 60:02}"

    def get_fill_ratio(self):
        return float(len(self.snake) / self.max_capacity * 100) if self.max_capacity > 0 else 0

    def _draw_glow_rect(self, color, rect, glow_size=6, radius=6):
        glow_surf = pygame.Surface((rect.width + glow_size, rect.height + glow_size), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*color, 60), glow_surf.get_rect(), border_radius=radius+2)
        self.display.blit(glow_surf, (rect.x - glow_size//2, rect.y - glow_size//2))
        pygame.draw.rect(self.display, color, rect, border_radius=radius)

    def _update_ui(self):
        self.display.fill(BLACK)
        for x in range(0, self.w, BLOCK_SIZE):
            pygame.draw.line(self.display, DARK_GRID, (x, 0), (x, self.h))
        for y in range(0, self.h, BLOCK_SIZE):
            pygame.draw.line(self.display, DARK_GRID, (0, y), (self.w, y))

        for wall in self.walls:
            rect = pygame.Rect(wall.x, wall.y, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(self.display, WALL_COLOR, rect)
            pygame.draw.rect(self.display, WALL_GLOW, rect, 2)

        level_color = SNAKE_COLORS[self.current_level - 1]
        for i, pt in enumerate(self.snake):
            fade = 1 - (i / len(self.snake) * 0.8)
            color = (int(level_color[0] * fade), int(level_color[1] * fade), int(level_color[2] * fade))
            rect = pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
            if i == 0:
                self._draw_glow_rect(level_color, rect, 8, 8)
            else:
                pygame.draw.rect(self.display, color, rect, border_radius=6)

        food_rect = pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE)
        self._draw_glow_rect((255, 80, 80), food_rect, 6, 10)

        if self.special_food:
            pulse = abs(math.sin(self.special_food_timer * 0.15))
            size = int(BLOCK_SIZE * (1 + pulse * 0.5))
            offset = (size - BLOCK_SIZE) // 2
            rect = pygame.Rect(self.special_food.x - offset, self.special_food.y - offset, size, size)
            self._draw_glow_rect(YELLOW, rect, 10, 12)

        hud = pygame.Surface((self.w, 40), pygame.SRCALPHA)
        hud.fill((10, 10, 20, 200))
        self.display.blit(hud, (0, 0))
        pygame.draw.line(self.display, (50, 50, 100), (0, 40), (self.w, 40))

        score_text = font.render(f"Lvl {self.current_level} | Score: {self.score}", True, WHITE)
        fill_text = font.render(f"Filled: {self.get_fill_ratio():.2f}%", True, level_color)
        time_text = font.render(f"{self._get_elapsed_time()}", True, WHITE)
        self.display.blit(score_text, (10, 8))
        self.display.blit(fill_text, (self.w // 2 - 70, 8))
        self.display.blit(time_text, (self.w - 100, 8))

        pygame.display.flip()