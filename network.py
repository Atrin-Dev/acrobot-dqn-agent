import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Q-Network
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, seed=42):
        super(DQN, self).__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
