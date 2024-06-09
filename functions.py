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
    
    # Normalize using sigmoid (add scaling to ensure the values are in a reasonable range)
    scaled_block = [sigmoid(price / 1000) for price in block]  # Example scaling by 1000
    scaled_capital = sigmoid(capital / 1000)
    
    state = np.array(scaled_block + [scaled_capital])
    
    # Debugging prints
    #print(f"Original block: {block}")
    #print(f"Scaled block: {scaled_block}")
    #print(f"Capital: {capital}, Scaled capital: {scaled_capital}")
    #print(f"State: {state}")
    
    return state