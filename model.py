import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import random
import numpy as np
from collections import deque

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward):
        state = torch.tensor(state, dtype=torch.float)
        pred = self.model(state)
        target = pred.clone()
        
        Q_new = reward
        target[action] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

def train_agent(model, target_model, episodes, gamma, epsilon, epsilon_decay, min_epsilon, lr, batch_size, replay_memory):
    trainer = QTrainer(model, lr, gamma)
    replay_buffer = deque(maxlen=replay_memory)
    scores = []

    for episode in range(episodes):
        green_pos = (random.randint(0, 5-1), random.randint(0, 5-1))
        state = np.zeros((5, 5))
        state[green_pos[1]][green_pos[0]] = 1
        state = state.flatten()
        done = False
        score = 0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            if random.uniform(0, 1) < epsilon:
                action_index = random.randint(0, 5 * 5 - 1)
            else:
                q_values = model(state_tensor)
                action_index = torch.argmax(q_values).item()

            action_x, action_y = get_coordinates(action_index)
            reward = -1
            if (action_x, action_y) == green_pos:
                reward = 1
                done = True

            score += reward
            
            trainer.train_step(state, action_index, reward)

        scores.append(score)
        print(f"Episode {episode + 1}: Score {score}")

        if epsilon > min_epsilon:
            epsilon *= epsilon_decay

        if episode % 10 == 0:
            target_model.load_state_dict(model.state_dict())

        if score >= 100:
            break

    return scores

def get_coordinates(state_index):
    return state_index % 5, state_index // 5

