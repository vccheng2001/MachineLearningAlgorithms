import sys
from environment import MountainCar
import pyglet
import numpy as np
import random 


def main():
    (program, mode, weight_out, returns_out, max_episodes, max_iterations, epsilon, gamma, learning_rate) = sys.argv
    # Initialize mountain car 
    car = MountainCar(mode, 1) # fixes at pos = 0.8, vel = 1
    # Set up
    num_actions, actions = 3, (0,1,2)
    weights = np.zeros((car.state_space, num_actions))
    num_episodes = 0
    # Do actions
    while num_episodes < max_episodes:
        num_episodes += 1
        # With prob epsilon, pick random action
        prob = random.random() 
        action = random.choice(action) if (prob < epsilon) else getBestAction()
        # Observe sample 
        (state, reward, done) = car.step(action)
        print(state, reward, done)
        if done:
            car.reset()
            car.close()
            return 
        else:
            update_weight()


def update_weight(action, weights, learning_rate, gamma, reward):
    sample = reward + gamma(maxQnext)
    currQ = np.dot(s, weights[:,action]) + b
    weights = weights - learning_rate*(currQ - sample)*gradient
    return weights 


def getBestAction():
    bestAction = None
    bestQ= float('-inf')
    for action in allActions:
        Q = self.getQValue(state,action)
        if Q >= bestQ:
        bestAction = action
        bestQ = Q 
    return bestAction 

if __name__ == "__main__":
    main()
