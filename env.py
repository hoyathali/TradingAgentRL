import gym
import numpy as np
from functions import *

class StockTradingEnv(gym.Env):
    def __init__(self, data, window_size):
        super(StockTradingEnv, self).__init__()
        self.data = data
        self.window_size = window_size
        self.action_space = gym.spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(window_size + 1,), dtype=np.float32)
        self.starting_cash = 10000  # Starting cash, adjust as needed
        self.max_holding_period = 50  # Maximum holding period for inventory items
        self.profit_reward_scale = 0.01  # Increased scaling factor for profit rewards
        self.loss_penalty_scale = 0.002  # Adjusted scaling factor for loss penalties
        self.holding_penalty_scale = 0.00005  # Further reduced scaling factor for holding penalties
        self.profit_bonus_scale = 0.005  # Increased scaling factor for final profit bonus

    def reset(self):
        self.current_step = 0
        self.inventory = []
        self.total_profit = 0
        self.total_cash = self.starting_cash
        self.cumulative_reward = 0
        self.total_holding_penalty = 0
        self.total_loss_penalty = 0
        return getState(self.data, self.current_step, self.window_size, self.total_cash)

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1 or self.current_step >= 7000  # End episode after 7000 steps
        reward = 0
        current_price = self.data[self.current_step]

        if action == 1 and self.total_cash >= current_price:  # Buy
            self.inventory.append((current_price, self.current_step))
            self.total_cash -= current_price

        elif action == 2 and self.inventory:  # Sell
            bought_price, bought_step = self.inventory.pop(0)
            profit = current_price - bought_price
            holding_time = min(self.current_step - bought_step, self.max_holding_period)
            profit_reward = profit * self.profit_reward_scale
            loss_penalty = abs(profit) * self.loss_penalty_scale if profit < 0 else 0
            holding_penalty = holding_time * self.holding_penalty_scale

            reward += profit_reward - loss_penalty - holding_penalty
            self.total_profit += profit
            self.total_cash += current_price
            self.total_loss_penalty += loss_penalty

        elif action == 0 and self.inventory:  # Hold
            reward -= self.holding_penalty_scale  # Penalty for holding without selling

        holding_penalty = sum(self.holding_penalty_scale * (min(self.current_step, len(self.data) - 1) - step) for _, step in self.inventory)
        reward -= holding_penalty
        self.total_holding_penalty += holding_penalty

        self.cumulative_reward += reward
        next_state = getState(self.data, self.current_step, self.window_size, self.total_cash)

        info = {
            'total_profit': self.total_profit,
            'holding_penalty': self.total_holding_penalty,
            'loss_penalty': self.total_loss_penalty,
            'cumulative_reward': self.cumulative_reward,
            'remaining_cash': self.total_cash
        }

        if done:
            # Apply a final profit bonus reward at the end of the episode
            final_profit_bonus = self.total_profit * self.profit_bonus_scale
            self.cumulative_reward += final_profit_bonus
            reward += final_profit_bonus

            print(f"Episode finished at step {self.current_step}. Total Reward: {self.cumulative_reward}, Total Profit: {self.total_profit}, Holding Penalties: {self.total_holding_penalty}, Loss Penalties: {self.total_loss_penalty}, Final Cash: {self.total_cash}")

        return next_state, reward, done, info
