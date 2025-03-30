import gymnasium as gym
import torch
import numpy as np
from collections import deque
from agent import DQNAgent

# Initialize the environment
env = gym.make('acrobot-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQNAgent(state_dim, action_dim)


def train(env_name, episodes, max_timesteps, epsilon_start, epsilon_end, epsilon_decay, model_path):
  env = gym.make(env_name)
  state_size = env.observation_space.shape[0]
  action_size = env.action_space.n
  agent = agent(state_size, action_size)

  epsilon = epsilon_start
  scores = deque(maxlen=100)

  for episode in range(1, episodes + 1):
    state, _ = env.reset()
    score = 0
    for t in range(max_timesteps):
      action = agent.select_action(state, epsilon)
      next_state, reward, done, _, _ = env.step(action)
      agent.step(state, action, reward, next_state, done)
      state = next_state
      score += reward
      if done:
        break

    scores.append(score)
    epsilon = max(epsilon_end, epsilon_decay * epsilon)
    print(f'\rEpisode {episode}\tAverage Score: {np.mean(scores):.2f}', end="")
    if episode % 100 == 0:
      print(f'\rEpisode {episode}\tAverage Score: {np.mean(scores):.2f}')
    if np.mean(scores) >= 200.0:
      print(f'\nEnvironment solved in {episode-100} episodes!\tAverage Score: {np.mean(scores):.2f}')
      torch.save(agent.local_model.state_dict(), model_path)
      break
