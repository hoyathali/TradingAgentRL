import gym
import numpy as np
import os
import torch
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from torch.utils.tensorboard import SummaryWriter
from env import *
from functions import *

class TensorboardCallback(BaseCallback):
    def __init__(self, writer, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.writer = writer
        self.episode_reward = 0

    def _on_step(self) -> bool:
        reward = self.locals['rewards'][0]
        self.episode_reward += reward
        self.writer.add_scalar('Reward', reward, self.num_timesteps)
        return True

    def _on_rollout_end(self) -> None:
        self.writer.add_scalar('Episode reward', self.episode_reward, self.num_timesteps)
        self.episode_reward = 0

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

stock_name = 'HSI'
window_size = 10
average_timesteps_per_episode = 7000
episode_count = 3000
total_timesteps = average_timesteps_per_episode * episode_count

# Get stock data and initialize environment
data = getStockDataVec(stock_name)
env = DummyVecEnv([lambda: StockTradingEnv(data, window_size)])

# TensorBoard writer setup
log_dir = f'runs/{stock_name}_window{window_size}_episodes{episode_count}_A2C'
writer = SummaryWriter(log_dir=log_dir)

# Checkpoint and Eval callbacks
checkpoint_callback = CheckpointCallback(save_freq=total_timesteps, save_path='./models/', name_prefix='a2c_model')
eval_callback = EvalCallback(env, best_model_save_path='./models/', log_path='./logs/', eval_freq=total_timesteps, deterministic=True, render=False)
tensorboard_callback = TensorboardCallback(writer)

# Check for existing checkpoints and load or initialize the model
model_path = './models/a2c_model_latest_ac.zip'
if os.path.exists(model_path):
    print("Loading model from checkpoint...")
    model = A2C.load(model_path, env=env, tensorboard_log=log_dir, device=device)
else:
    print("No checkpoint found, initializing new model...")
    model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=log_dir, device=device)

# Training the model
model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback, tensorboard_callback])

# Save the final model
model.save("models/final_a2c_model")
# Close the TensorBoard writer
writer.close()
# Save the latest model checkpoint
model.save(model_path)
