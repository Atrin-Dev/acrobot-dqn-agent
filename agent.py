import torch
import random
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from network import Network
from memory import ReplayBuffer

# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=5e-4, gamma=0.99, buffer_size=int(1e5), tau=1e-3):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.local_model = DQN(state_dim, action_dim).to(self.device)
        self.target_model = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.local_model.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        self.gamma = gamma
        self.tau = tau
        self.timestep = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.store((state, action, reward, next_state, done))
        self.timestep = (self.timestep + 1) % 4
        if self.timestep == 0 and len(self.memory.memory) > 100:
            experiences = self.memory.sample_batch(100)
            self.learn(experiences)

    def select_action(self, state, epsilon=0.0):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.local_model.eval()
        with torch.no_grad():
            action_values = self.local_model(state)
        self.local_model.train()
        return np.argmax(action_values.cpu().data.numpy()) if random.random() > epsilon else random.choice(np.arange(self.action_dim))

    def learn(self, experiences):
        states, next_states, actions, rewards, dones = experiences
        next_q_targets = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + self.gamma * next_q_targets * (1 - dones)
        q_expected = self.local_model(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update()

    def soft_update(self):
        for target_param, local_param in zip(self.target_model.parameters(), self.local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
