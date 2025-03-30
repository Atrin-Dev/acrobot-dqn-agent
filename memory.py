import random
import torch
import numpy as np

# Define the Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.capacity = capacity
        self.memory = []

    def store(self, experience):
        self.memory.append(experience)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample_batch(self, batch_size):
        experiences = random.sample(self.memory, batch_size)
        states = torch.tensor(np.vstack([e[0] for e in experiences]), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.vstack([e[1] for e in experiences]), dtype=torch.int64).to(self.device)
        rewards = torch.tensor(np.vstack([e[2] for e in experiences]), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.vstack([e[3] for e in experiences]), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.vstack([e[4] for e in experiences]).astype(np.uint8), dtype=torch.float32).to(self.device)
        return states, next_states, actions, rewards, dones
