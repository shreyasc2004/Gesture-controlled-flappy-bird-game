import pygame
import sys
import random

# --- Constants ---
WIDTH, HEIGHT = 400, 600
FPS = 60
GRAVITY = 0.25
FLAP_STRENGTH = -6
PIPE_GAP = 200
PIPE_FREQUENCY = 3000 # milliseconds <-- Increased from 2500 for more space
PIPE_SPEED = 3

# --- Colors ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
SKY_BLUE = (135, 206, 235)
GREEN = (0, 200, 0)
DARK_GREEN = (0, 100, 0)

# --- Bird Class ---
class Bird(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((30, 30), pygame.SRCALPHA)
        pygame.draw.circle(self.image, (255, 255, 0), (15, 15), 15) # Yellow circle
        self.rect = self.image.get_rect(center=(100, HEIGHT // 2))
        self.velocity = 0

    def update(self, action):
        # Apply gravity
        self.velocity += GRAVITY
        
        # Apply flap or dive
        if action == "THUMBS_UP": 
            self.velocity = FLAP_STRENGTH
        elif action == "THUMBS_DOWN": 
            self.velocity += 1.0 # Accelerate downwards
        
        # Update position
        self.rect.y += self.velocity

        # Keep bird on screen (top)
        if self.rect.top < 0:
            self.rect.top = 0
            self.velocity = 0

# --- Pipe Class ---
class Pipe(pygame.sprite.Sprite):
    def __init__(self, x, y, is_top):
        super().__init__()
        self.image = pygame.Surface((50, 400))
        self.image.fill(GREEN)
        pygame.draw.rect(self.image, DARK_GREEN, self.image.get_rect(), 5) # Border
        self.rect = self.image.get_rect()
        if is_top:
            self.rect.bottomleft = (x, y - PIPE_GAP // 2)
        else:
            self.rect.topleft = (x, y + PIPE_GAP // 2)

    def update(self):
        self.rect.x -= PIPE_SPEED
        if self.rect.right < 0:
            self.kill()

# --- Game Class ---
class FlappyBirdGame:
    def __init__(self):
        pygame.init()
        pygame.font.init() # Initialize font module
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Gesture Flappy Bird")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 50) # Default font, size 50
        
        self.bird_group = pygame.sprite.GroupSingle()
        self.bird = Bird()
        self.bird_group.add(self.bird)
        
        self.pipe_group = pygame.sprite.Group()
        
        self.score = 0
        self.game_started = False
        self.game_over = False
        self.running = True 
        self.last_pipe_time = pygame.time.get_ticks()

    def _create_pipe_pair(self):
        pipe_y = random.randint(HEIGHT // 4, 3 * HEIGHT // 4)
        top_pipe = Pipe(WIDTH, pipe_y, is_top=True)
        bottom_pipe = Pipe(WIDTH, pipe_y, is_top=False)
        self.pipe_group.add(top_pipe, bottom_pipe)

    def _show_start_screen(self):
        self.screen.fill(SKY_BLUE)
        self._draw_text("Gesture Flappy Bird", 30, WIDTH // 2, HEIGHT // 3)
        self._draw_text("Show THUMBS UP to Start!", 25, WIDTH // 2, HEIGHT // 2)
        self._draw_text("(Up = Flap, Down = Dive)", 20, WIDTH // 2, HEIGHT // 2 + 40)
        pygame.display.flip()

    def _show_game_over_screen(self):
        self.screen.fill(SKY_BLUE)
        self._draw_text("Game Over", 50, WIDTH // 2, HEIGHT // 3)
        self._draw_text(f"Score: {self.score}", 40, WIDTH // 2, HEIGHT // 2)
        self._draw_text("Show THUMBS UP to Restart", 25, WIDTH // 2, HEIGHT // 2 + 50)
        pygame.display.flip()

    def _draw_text(self, text, size, x, y):
        font = pygame.font.Font(None, size)
        text_surface = font.render(text, True, BLACK)
        text_rect = text_surface.get_rect(center=(x, y))
        self.screen.blit(text_surface, text_rect)

    def _reset_game(self):
        self.bird.rect.center = (100, HEIGHT // 2)
        self.bird.velocity = 0
        self.pipe_group.empty()
        self.score = 0
        self.game_started = True
        self.game_over = False
        self.last_pipe_time = pygame.time.get_ticks()

    def update(self, action="NEUTRAL"):
        # Handle Pygame events (like closing the window)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False 
                return # Immediately return to stop processing
        
        # --- Game State Machine ---
        if not self.running:
            return

        if self.game_over:
            if action == "THUMBS_UP": 
                self._reset_game()
        elif not self.game_started:
            if action == "THUMBS_UP": 
                self.game_started = True
                self.last_pipe_time = pygame.time.get_ticks()
        else:
            # --- Game is running ---
            
            # 1. Update Bird
            self.bird_group.update(action)
            
            # 2. Update Pipes
            self.pipe_group.update()
            
            # 3. Spawn new pipes
            now = pygame.time.get_ticks()
            if now - self.last_pipe_time > PIPE_FREQUENCY:
                self.last_pipe_time = now
                self._create_pipe_pair()
            
            # 4. Check for collisions
            if pygame.sprite.spritecollide(self.bird, self.pipe_group, False) or \
               self.bird.rect.bottom >= HEIGHT:
                self.game_over = True
            
            # 5. Update Score
            # Find pipes that have passed the bird and haven't been scored
            passed_pipes = []
            for pipe in self.pipe_group:
                if pipe.rect.right < self.bird.rect.left and not hasattr(pipe, 'scored'):
                    passed_pipes.append(pipe)
                    pipe.scored = True # Mark as scored to prevent double counting
            
            # Add 0.5 for each pipe (top and bottom), resulting in +1 per pair
            self.score += len(passed_pipes) * 0.5


    def draw(self):
        if not self.running:
            return

        self.screen.fill(SKY_BLUE)
        
        if self.game_over:
            self._show_game_over_screen()
        elif not self.game_started:
            self._show_start_screen()
        else:
            # Draw game elements
            self.bird_group.draw(self.screen)
            self.pipe_group.draw(self.screen)
            
            # Draw Score
            score_text = self.font.render(str(int(self.score)), True, WHITE)
            self.screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, 50))
        
        pygame.display.flip()
        self.clock.tick(FPS)

    def quit(self):
        pygame.quit()

