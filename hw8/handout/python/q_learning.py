import sys
from environment import MountainCar
import pyglet
import numpy as np
import random 


def main():
    (program, mode, weight_out, returns_out, episodes, max_iterations, epsilon, gamma, learning_rate) = sys.argv
    # Initialize mountain car 
    car = MountainCar(mode, 1) # fixes at pos = 0.8, vel = 1
    # Set up
    num_actions, actions = 3, (0,1,2)
    weights = np.zeros((car.state_space, num_actions))
    # Do actions
    while True:
        prob = random.random()
        if (prob <= epsilon):
             action = random.choice(actions)
        else: 
            # choose best 
    # For each action
    for i in range(num_actions):
        action = actions[i] 
        q[s][a] = np.dot(s, weights[:,action]) + b
        # Observe sample
        (state, reward, done) = car.step(action)
        pos, vel = state 
    # 
    print(state, reward, done)
    if done:
        car.reset()
        car.close()


def weightUpdate(learning_rate, gamma, reward):
    sample = reward + gamma(max...)

    weights = weights - learning_rate()

    return weights 
if __name__ == "__main__":
    main()
