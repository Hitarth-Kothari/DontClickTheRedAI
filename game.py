import pygame
import random
import numpy as np
import sys
import matplotlib.pyplot as plt
from model import Linear_QNet, train_agent, get_coordinates
import torch

# Initialize Pygame
pygame.init()

# Screen dimensions
width, height = 500, 500
rows, cols = 5, 5
cell_size = width // cols

# Colors
red = (255, 0, 0)
green = (0, 255, 0)
white = (255, 255, 255)
black = (0, 0, 0)

# Create screen
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Find the Green Square")

# Font
font = pygame.font.SysFont(None, 55)
small_font = pygame.font.SysFont(None, 35)

# Function to get the state index from coordinates
def get_state_index(x, y):
    return y * cols + x

# Function to draw the grid
def draw_grid(state):
    for x in range(0, width, cell_size):
        for y in range(0, height, cell_size):
            rect = pygame.Rect(x, y, cell_size, cell_size)
            pygame.draw.rect(screen, black, rect, 1)
            if state[y // cell_size][x // cell_size] == 1:
                pygame.draw.rect(screen, green, rect)
            else:
                pygame.draw.rect(screen, red, rect)

# Function to draw the score
def draw_text(score):
    score_text = small_font.render(f"Score: {score}", True, black)
    screen.blit(score_text, (10, 10))

# Function to end the game
def game_over(score):
    screen.fill(white)
    game_over_text = font.render(f"Game Over! Score: {score}", True, black)
    screen.blit(game_over_text, (50, height // 2 - 25))
    pygame.display.flip()
    pygame.time.wait(3000)
    pygame.quit()
    sys.exit()

# Function to test the agent
def test_agent(model):
    green_pos = (random.randint(0, cols-1), random.randint(0, rows-1))
    state = np.zeros((5, 5))
    state[green_pos[1]][green_pos[0]] = 1
    state = state.flatten()
    score = 0

    running = True
    while running:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = model(state_tensor)
            action_index = torch.argmax(q_values).item()

        action_x, action_y = get_coordinates(action_index)

        if (action_x, action_y) == green_pos:
            score += 1
            if score >= 100:
                print("Test Ended with Score of : ", score)
                game_over(score)
                running = False
            green_pos = (random.randint(0, cols-1), random.randint(0, rows-1))
            state = np.zeros((5, 5))
            state[green_pos[1]][green_pos[0]] = 1
            state = state.flatten()
        else:
            print("Test Ended with Score of : ", score)
            game_over(score)
            running = False

if __name__ == "__main__":
    input_size = 5 * 5
    hidden_size = 64
    output_size = 5 * 5
    model = Linear_QNet(input_size, hidden_size, output_size)
    target_model = Linear_QNet(input_size, hidden_size, output_size)
    target_model.load_state_dict(model.state_dict())

    episodes = 1000
    gamma = 0.9
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.01
    lr = 0.001
    batch_size = 64
    replay_memory = 10000
    scores = train_agent(model, target_model, episodes, gamma, epsilon, epsilon_decay, min_epsilon, lr, batch_size, replay_memory)
    
    # Plot the scores
    plt.plot(scores)
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.title('Performance over Time')
    plt.show()
    
    # Test the agent
    test_agent(model)
