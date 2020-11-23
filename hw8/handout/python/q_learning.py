import sys
from environment import MountainCar
import pyglet
import numpy as np


def main():
    (program, mode, weight_out, returns_out, episodes, max_iterations, epsilon, gamma, learning_rate) = sys.argv
    
    car = MountainCar(mode, 1) # fixes at pos = 0.8, vel = 1
    A, actions = 3, (0,1,2)
    S = 2048 if mode == "tile" else 2
    weights = np.zeros((S, A))
    (state, reward, done) = car.step(0)
    pos, vel = state 
    print(state, reward, done)
    if done:
        car.reset()
        car.close()


if __name__ == "__main__":
    main()
