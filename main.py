import argparse
from train import train
from visualize import visualize, show_video

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Train and visualize a DQN agent on acrobot-v1")
  parser.add_argument("--env", type=str, default="acrobot-v1", help="Gym environment name")
  parser.add_argument("--episodes", type=int, default=2000, help="Number of episodes")
  parser.add_argument("--timesteps", type=int, default=1000, help="Max timesteps per episode")
  parser.add_argument("--epsilon_start", type=float, default=1.0, help="Initial epsilon value")
  parser.add_argument("--epsilon_end", type=float, default=0.01, help="Minimum epsilon value")
  parser.add_argument("--epsilon_decay", type=float, default=0.995, help="Epsilon decay rate")
  parser.add_argument("--model_path", type=str, default="checkpoint.pth", help="Path to save/load the trained model")
  parser.add_argument("--visualize", action="store_true", help="Visualize the trained agent")
  parser.add_argument("--video_path", type=str, default="video.mp4", help="Path to save the video output")

  args = parser.parse_args()

  if args.visualize:
    visualize(args.model_path, args.env, args.video_path)
    show_video(args.video_path)
  else:
    train(args.env, args.episodes, args.timesteps, args.epsilon_start, args.epsilon_end, args.epsilon_decay,
          args.model_path)
