import os
import pygame
import random
from enum import Enum
from collections import namedtuple
import math
import time

pygame.init()

# ----- Constants -----
BLOCK_SIZE = 20
SPEED_BASE = 8  # Slightly faster base speed for better feel

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (200, 0, 0)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
DARK_GRID = (15, 15, 25)
WALL_COLOR = (80, 80, 100)
WALL_GLOW = (40, 40, 60)

# Snake Color Evolution (Per Level)
SNAKE_COLORS = [
    (0, 180, 255),   # Level 1: Neon Blue
    (0, 255, 120),   # Level 2: Neon Green
    (160, 80, 255),  # Level 3: Purple
    (255, 140, 60),  # Level 4: Orange
    (255, 215, 0),   # Level 5: Gold
]

# Font
font = pygame.font.Font('Default.ttf', 20)
large_font = pygame.font.Font('Default.ttf', 40)

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

# ----- Helper Classes -----
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# ----- Snake Game -----
class SnakeGame:

    def __init__(self):
        self.clock = pygame.time.Clock()
        self.start_time = time.time()
        self.high_score = self._load_high_score()
        
        # Game State
        self.current_level = 1
        self.max_levels = 5
        self.score = 0
        self.level_score = 0  # Score accumulated specifically in this level
        
        # Determine score needed to pass level (scales with difficulty)
        self.score_to_next_level = 5 
        
        self.paused = False
        self.game_won = False
        
        # Initialize First Level
        self._init_level_properties(self.current_level)

    def _init_level_properties(self, level):
        # 1. Screen Zoom / Resize Logic
        base_w, base_h = 400, 400 
        extra = (level - 1) * 100 
        
        self.w = base_w + extra
        self.h = base_h + extra
        
        # Re-initialize display for new size
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption(f'Snake Game - Level {level}')
        
        # 2. Reset Snake Position (Center)
        start_x = (self.w // 2) // BLOCK_SIZE * BLOCK_SIZE
        start_y = (self.h // 2) // BLOCK_SIZE * BLOCK_SIZE
        self.head = Point(start_x, start_y)
        self.direction = Direction.RIGHT
        self.next_direction = self.direction
        
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2 * BLOCK_SIZE, self.head.y)
        ]

        # 3. Generate Walls
        self.walls = []
        self._generate_walls(level)
        
        # 4. Calculate Max Capacity (Total Slots - Wall Slots)
        total_cols = self.w // BLOCK_SIZE
        total_rows = self.h // BLOCK_SIZE
        total_slots = total_cols * total_rows
        wall_slots = len(self.walls)
        
        # The level is finished when the snake fills every non-wall spot
        self.max_capacity = total_slots - wall_slots

        # 5. Place Food
        self.food = None
        self.special_food = None
        self.special_food_timer = 0
        self._place_food()

    def _generate_walls(self, level):
        """Generates wall layout based on level"""
        self.walls = []
        cols = self.w // BLOCK_SIZE
        rows = self.h // BLOCK_SIZE

        if level == 1:
            pass # Open field
            
        elif level == 2:
            # Four Pillars near corners
            for x in [5, cols-6]:
                for y in [5, rows-6]:
                    self.walls.append(Point(x*BLOCK_SIZE, y*BLOCK_SIZE))
                    
        elif level == 3:
            # Horizontal Bars
            y_positions = [rows//3, (rows//3)*2]
            for y in y_positions:
                for x in range(4, cols-4):
                    self.walls.append(Point(x*BLOCK_SIZE, y*BLOCK_SIZE))

        elif level == 4:
            # The Box (Center obstacle)
            center_x = cols // 2
            center_y = rows // 2
            for x in range(center_x - 5, center_x + 5):
                for y in range(center_y - 5, center_y + 5):
                    # Hollow box
                    if x == center_x-5 or x == center_x+4 or y == center_y-5 or y == center_y+4:
                        self.walls.append(Point(x*BLOCK_SIZE, y*BLOCK_SIZE))

        elif level == 5:
            # The Maze (Randomized but symmetrical)
            random.seed(42) # Fixed seed so the maze is learnable per run
            for x in range(0, cols, 2):
                for y in range(0, rows, 2):
                    if random.random() > 0.7:
                        # Don't spawn walls too close to center (snake spawn)
                        if abs(x - cols//2) > 4 or abs(y - rows//2) > 4:
                            self.walls.append(Point(x*BLOCK_SIZE, y*BLOCK_SIZE))
                            # Add a neighbor to make it wall-like
                            self.walls.append(Point((x+1)*BLOCK_SIZE, y*BLOCK_SIZE))

    # ----- Food placement -----
    def _place_food(self):
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            candidate = Point(x, y)
            
            # Ensure food doesn't spawn in snake, special food, or WALLS
            if (candidate not in self.snake and 
                candidate != self.special_food and 
                candidate not in self.walls):
                self.food = candidate
                break

    def _place_special_food(self):
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            candidate = Point(x, y)
            if (candidate not in self.snake and 
                candidate != self.food and 
                candidate not in self.walls):
                self.special_food = candidate
                self.special_food_timer = 0
                if SPECIAL_FOOD_SOUND: SPECIAL_FOOD_SOUND.play()
                break

    # ----- High Score -----
    def _load_high_score(self):
        if os.path.exists("highscore.txt"):
            try:
                with open("highscore.txt", "r") as f:
                    return int(f.read())
            except: return 0
        return 0

    def _save_high_score(self):
        with open("highscore.txt", "w") as f:
            f.write(str(self.high_score))

    # ----- Game Step -----
    def play_step(self):
        self._handle_events()
        
        if self.game_won:
            self._draw_victory()
            return False, self.score

        if not self.paused:
            self.direction = self.next_direction
            self._move(self.direction)
            self.snake.insert(0, self.head)

            game_over = False
            
            # Check Collision (Walls or Self)
            if self._is_collision():
                if GAME_OVER_SOUND: GAME_OVER_SOUND.play()
                game_over = True
                if self.score > self.high_score:
                    self.high_score = self.score
                    self._save_high_score()
                return game_over, self.score

            # Check Food
            eaten = False
            if self.head == self.food:
                self.score += 1
                eaten = True
                if EAT_SOUND: EAT_SOUND.play()
                # Only place new food if screen isn't full
                if len(self.snake) < self.max_capacity:
                    self._place_food()
                
            elif self.head == self.special_food:
                self.score += 3
                eaten = True
                if EAT_SOUND: EAT_SOUND.play()
                self.special_food = None
            else:
                # If we didn't eat, remove tail
                self.snake.pop()

            # Special food logic
            if self.score > 0 and self.score % 10 == 0 and self.special_food is None and not eaten:
                 # Don't place special food if it might block the final path too much
                 if len(self.snake) < self.max_capacity - 5:
                    self._place_special_food()
            
            if self.special_food:
                self.special_food_timer += 1
                if self.special_food_timer > 100:
                    self.special_food = None

            # --- UPDATED LEVEL UP LOGIC ---
            # Condition: Snake length equals available space (Screen Filled)
            if len(self.snake) >= self.max_capacity:
                if self.current_level < self.max_levels:
                    self.current_level += 1
                    if LEVEL_UP_SOUND: LEVEL_UP_SOUND.play()
                    self._init_level_properties(self.current_level)
                    time.sleep(0.5)
                else:
                    self.game_won = True
                    if GAME_COMPLETE: GAME_COMPLETE.play()

            # Speed scales with snake length (get faster as you get larger)
            # Cap the speed so it doesn't become impossible when full
            base_speed_boost = (self.current_level - 1) * 2
            length_factor = len(self.snake) * 0.1
            speed = min(SPEED_BASE + base_speed_boost + length_factor, 25) 
            self.clock.tick(speed)

        # Update UI
        self._update_ui()
        return False, self.score

    # ----- Move Snake -----
    def _move(self, direction):
        x, y = self.head.x, self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)

    # ----- Collision Detection -----
    def _is_collision(self):
        # 1. Boundary Check
        if self.head.x >= self.w or self.head.x < 0 or self.head.y >= self.h or self.head.y < 0:
            return True
        # 2. Self Check
        if self.head in self.snake[1:]:
            return True
        # 3. Wall Check
        if self.head in self.walls:
            return True
            
        return False

    # ----- Input Handling -----
    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    self.paused = not self.paused
                
                # Restart if game won
                if self.game_won and event.key == pygame.K_SPACE:
                     self.__init__() # Reset game

                if not self.paused and not self.game_won:
                    if (event.key == pygame.K_LEFT or event.key == pygame.K_a) and self.direction != Direction.RIGHT:
                        self.next_direction = Direction.LEFT
                    elif (event.key == pygame.K_RIGHT or event.key == pygame.K_d) and self.direction != Direction.LEFT:
                        self.next_direction = Direction.RIGHT
                    elif (event.key == pygame.K_UP or event.key == pygame.K_w) and self.direction != Direction.DOWN:
                        self.next_direction = Direction.UP
                    elif (event.key == pygame.K_DOWN or event.key == pygame.K_s) and self.direction != Direction.UP:
                        self.next_direction = Direction.DOWN

    def _get_elapsed_time(self):
        elapsed = int(time.time() - self.start_time)
        minutes = elapsed // 60
        seconds = elapsed % 60
        return f"{minutes:02}:{seconds:02}"
    
    def _draw_glow_rect(self, color, rect, glow_size=6, radius=6):
        # Simplified glow for performance
        glow_surf = pygame.Surface((rect.width + glow_size, rect.height + glow_size), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*color, 60), glow_surf.get_rect(), border_radius=radius+2)
        self.display.blit(glow_surf, (rect.x - glow_size//2, rect.y - glow_size//2))
        pygame.draw.rect(self.display, color, rect, border_radius=radius)

    # ----- Render -----
    def _update_ui(self):
        self.display.fill(BLACK)

        # Draw Grid
        for x in range(0, self.w, BLOCK_SIZE):
            pygame.draw.line(self.display, DARK_GRID, (x, 0), (x, self.h))
        for y in range(0, self.h, BLOCK_SIZE):
            pygame.draw.line(self.display, DARK_GRID, (0, y), (self.w, y))

        # Draw Walls
        for wall in self.walls:
            rect = pygame.Rect(wall.x, wall.y, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(self.display, WALL_COLOR, rect)
            pygame.draw.rect(self.display, WALL_GLOW, rect, 2) # Border

        # Get Current Level Color
        level_color = SNAKE_COLORS[self.current_level - 1]

        # Draw snake with dynamic color fade
        length = len(self.snake)
        for i, pt in enumerate(self.snake):
            # Calculate fade factor (1.0 at head, 0.2 at tail)
            fade_factor = 1 - (i / length * 0.8) 
            
            # Apply fade to the specific level color
            curr_r = int(level_color[0] * fade_factor)
            curr_g = int(level_color[1] * fade_factor)
            curr_b = int(level_color[2] * fade_factor)
            body_color = (curr_r, curr_g, curr_b)

            rect = pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)

            if i == 0:
                self._draw_glow_rect(level_color, rect, glow_size=8, radius=8)
            else:
                pygame.draw.rect(self.display, body_color, rect, border_radius=6)

        # Draw Food
        food_rect = pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE)
        self._draw_glow_rect((255, 80, 80), food_rect, glow_size=6, radius=10)

        # Draw Special Food
        if self.special_food:
            pulse = abs(math.sin(self.special_food_timer * 0.15))
            size = int(BLOCK_SIZE * (1 + pulse * 0.5))
            offset = (size - BLOCK_SIZE) // 2
            core_rect = pygame.Rect(self.special_food.x - offset, self.special_food.y - offset, size, size)
            self._draw_glow_rect(YELLOW, core_rect, glow_size=10, radius=12)

        # HUD
        hud_height = 40
        hud_surface = pygame.Surface((self.w, hud_height), pygame.SRCALPHA)
        
        # Fill it with a color including Alpha (R, G, B, Alpha)
        hud_surface.fill((10, 10, 20, 150)) 
        
        self.display.blit(hud_surface, (0, 0))
        pygame.draw.line(self.display, (50, 50, 100), (0, hud_height), (self.w, hud_height))

        # Calculate % Filled
        percent_filled = float((len(self.snake) / self.max_capacity) * 100)

        # Render Text on top
        score_text = font.render(f"Lvl {self.current_level} | Score: {self.score}", True, WHITE)
        next_lvl_text = font.render(f"Filled: {percent_filled:.2f}%", True, level_color)
        time_text = font.render(f"{self._get_elapsed_time()}", True, WHITE)
        
        self.display.blit(score_text, [10, 8])
        self.display.blit(next_lvl_text, [self.w // 2 - 50, 8])
        self.display.blit(time_text, [self.w - 70, 8])
        # ---------------------------

        if self.paused:
            pause_text = font.render("PAUSED", True, YELLOW)
            self.display.blit(pause_text, [self.w//2 - 40, self.h//2])
        
        pygame.display.flip()

    def _draw_victory(self):
        self.display.fill(BLACK)
        
        title = large_font.render("VICTORY!", True, SNAKE_COLORS[4]) # Gold
        score_msg = font.render(f"Final Score: {self.score}", True, WHITE)
        time_msg = font.render(f"Total Time: {self._get_elapsed_time()}", True, WHITE)
        hint = font.render("Press SPACE to Restart", True, (150, 150, 150))
        
        cx, cy = self.w // 2, self.h // 2
        
        self.display.blit(title, (cx - title.get_width()//2, cy - 80))
        self.display.blit(score_msg, (cx - score_msg.get_width()//2, cy - 20))
        self.display.blit(time_msg, (cx - time_msg.get_width()//2, cy + 10))
        self.display.blit(hint, (cx - hint.get_width()//2, cy + 60))
        
        pygame.display.flip()

# ----- Main Loop -----
if __name__ == '__main__':
    game = SnakeGame()

    while True:
        game_over, score = game.play_step()
        if game_over:
            # Simple Game Over Delay
            time.sleep(1)
            break

    print('Final Score', score)
    pygame.quit()