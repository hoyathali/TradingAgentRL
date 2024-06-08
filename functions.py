import gym
import numpy as np
from functions import *

def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def getStockDataVec(key):
    vec = []
    try:
        lines = open(f"data/{key}.csv", "r").read().splitlines()
        for line in lines[1:]:  # Skip header
            parts = line.split(',')
            if parts[4] != 'null':  # Assuming 'Close' prices are in the fifth column
                vec.append(float(parts[4]))  # Convert to float and append to the list
    except FileNotFoundError:
        print("File not found. Please check the filename and path.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return vec

def getState(data, t, n, capital):
    d = t - n + 1
    block = data[max(0, d):t + 1] if d >= 0 else [data[0]] * (-d) + data[:t + 1]
    block = [sigmoid(price) for price in block]  # Normalize using sigmoid
    state = np.array(block + [sigmoid(capital)])  # Include capital in the state
    return state
