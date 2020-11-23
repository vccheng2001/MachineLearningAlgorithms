import sys
from environment import MountainCar
import pyglet
import numpy as np
import random 


def main():
    (program, mode, weight_out, returns_out, episodes, max_iterations, epsilon, gamma, learning_rate) = sys.argv
    epsilon, episodes, max_iterations = float(epsilon), int(episodes), int(max_iterations)
    # Initialize mountain car 
    car = MountainCar(mode, 1) # fixes at pos = 0.8, vel = 1
    # Set up
    num_actions, actions = 3, (0,1,2)
    weights = np.zeros((car.state_space, num_actions))
    # Do actions
    for episode in range(episodes):
        num_iters = 0
        while num_iters < max_iterations:
            num_iters += 1
            # With prob epsilon, pick random action
            prob = random.random() 
            action = random.choice(action) if (prob < epsilon) else getBestAction()
            # Observe sample 
            (state, reward, done) = car.step(action)
            (pos, vel) = state
            print(state, reward, done)
            if done:
                print('done')
                car.reset()
                return 
            else:
                print(state, reward, done)
                #update_weight()
        car.reset()

    car.close()


# def update_weight(action, weights, learning_rate, gamma, reward):
#     sample = reward + gamma(maxQnext)
#     currQ = np.dot(s, weights[:,action]) + b
#     gradient = s
#     weights = weights - learning_rate*(currQ - sample)*gradient
#     return weights 

# Return Q value given state, action
def getQValue(Q, state, action):
    return Q[state][action] 

# Return best action for given state 
def getBestAction(Q, state, allActions):
    bestAction = None
    bestQ= float('-inf')
    for action in allActions:
        currQ = getQValue(Q, state,action)
        if currQ >= bestQ:
            bestAction = action
            bestQ = currQ
    return bestAction 

if __name__ == "__main__":
    main()
