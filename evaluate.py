import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from env import StockTradingEnv
from functions import getStockDataVec

# Load the model
model_path = 'models/final_dqn_model.zip'
model = DQN.load(model_path)
window_size=10
# Load the dataset for the year 2018
stock_name = 'HSI_2018'
data_2018 = getStockDataVec(stock_name)
env = DummyVecEnv([lambda: StockTradingEnv(data_2018, window_size)])

# Initialize the environment for evaluation
obs = env.reset()
done = False
total_rewards = []
rewards = []

# Run the evaluation
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    rewards.append(reward)
    print(obs)
    print(action)
    if done:
        total_rewards.append(sum(rewards))
        obs = env.reset()  # reset for the next evaluation if needed
        rewards = []

# Plotting the rewards
plt.figure(figsize=(10, 5))
plt.plot(total_rewards)
plt.xlabel('Episodes')
plt.ylabel('Total Rewards')
plt.title('Performance of DQN on HSI 2018 Data')
plt.grid(True)

# Save the plot as a PNG file
plt.savefig('DQN_HSI_2018_Performance.png')

# Optionally, you can close the plot if not showing it
plt.close()