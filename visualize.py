import gymnasium as gym
import torch
import imageio
import base64
import io
from IPython.display import HTML, display
from agent import DQNAgent


# Visualization

def visualize(agent_path, env_name, output_video):
  env = gym.make(env_name, render_mode='rgb_array')
  state, _ = env.reset()
  agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
  agent.local_qnetwork.load_state_dict(torch.load(agent_path))
  done = False
  frames = []

  while not done:
    frame = env.render()
    frames.append(frame)
    action = agent.act(state)
    state, reward, done, _, _ = env.step(action.item())

  env.close()
  imageio.mimsave(output_video, frames, fps=30)
  print(f'Video saved as {output_video}')


def show_video(video_path):
  video = io.open(video_path, 'r+b').read()
  encoded = base64.b64encode(video).decode('ascii')
  display(HTML(
    f'<video autoplay loop controls style="height: 400px;"><source src="data:video/mp4;base64,{encoded}" type="video/mp4" /></video>'))
