# DontClickTheRedAI

DontClickTheRedAI is a simple game where the player needs to find the green square on a 5x5 grid. This game also includes an AI agent using reinforcement learning to play and learn the game.

## Game Description

- The grid consists of 5x5 squares.
- A green square is randomly spawned on the grid while all others are red.
- The player needs to input the coordinates of the green square.
- If the input is correct, the score increases by 1, otherwise, the game ends.
- The objective is to reach a score of 100.

## AI Agent

The AI agent uses a neural network to learn and play the game. It is trained using reinforcement learning. The agent is rewarded for finding the green square and penalized otherwise.

## How to Run

### Prerequisites

- Python 3.x
- Pygame
- NumPy
- Matplotlib
- PyTorch

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/DontClickTheRedAI.git
   cd DontClickTheRedAI
2. Install the required packages:
    ```bash
    pip install -r requirements.txt

### Training the AI

1. Run the training script:
   ```bash
   python game.py
2. The training process will run for a specified number of episodes. The scores will be plotted to show the performance over time.

### Testing the AI

1. After training, the AI agent will automatically play the game.
2. The game will end when the agent reaches a score of 100 or makes a wrong move.