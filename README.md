# Deep Q-Learning for Acrobot-v1

## 🚀 Introduction
This repository implements **Deep Q-Learning (DQN)** for training an agent to solve the *Acrobot-v1* environment from OpenAI Gymnasium.

## 📌 Features
- **Implements Deep Q-Network (DQN) with Experience Replay**
- **Uses PyTorch for neural network training**
- **Supports GPU acceleration (if available)**
- **Provides visualization of the trained agent**

## 📂 Project Structure
```plaintext
dqn-acrobot-v1
│── 📜 README.md           # Project documentation
│── 📜 requirements.txt     # Dependencies
│── 📜 train.py            # Training script
│── 📜 test.py             # Evaluation script
│── 📜 model.py            # Neural network architecture
│── 📜 agent.py            # DQN agent implementation
│── 📜 replay_buffer.py    # Experience replay buffer
│── 📜 environment.py      # Gym environment setup
│── 📜 visualize.py        # Video visualization script
```
###  Install Dependencies
```bash
pip install -r requirements.txt
```
### 🎮 Training the agent
Run the following command to train the DQN agent:
```bash
python train.py --episodes 3000 --timesteps 1500 --epsilon_start 1.0 --epsilon_end 0.05 --epsilon_decay 0.99 --model_path "trained_model.pth"
```
This will train the agent in the Acrobot-v1 environment and save the model checkpoint if the agent solves the environment.
### 📊 Visualizing the trained agent
To visualize the trained agent, run:
```bash
python script.py --visualize --model_path checkpoint.pth --video_path video.mp4
```
This will generate a video of the agent interacting with the environment.

### 📊 Visualizing the Results
[Watch the trained agent by clicking the link](https://github.com/Atrin-Dev/acrobot-dqn-agent/blob/main/Acrobot.gif)
![Demo](https://github.com/Atrin-Dev/acrobot-dqn-agent/blob/main/Acrobot.gif)

### 📝 Configuration

You can modify the hyperparameters in main.py:
```
learning_rate = 5e-4
batch_size = 100
discount_factor = 0.99
memory_size = int(1e5)
target_update_rate = 1e-3
```
### 📢 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.
