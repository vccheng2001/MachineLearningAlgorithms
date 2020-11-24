import sys
from environment import MountainCar
import pyglet
import numpy as np
import random 


def main():
    (program, mode, weight_out, returns_out, episodes, max_iterations, epsilon, gamma, alpha) = sys.argv
    epsilon, episodes, max_iterations = float(epsilon), int(episodes), int(max_iterations)
    # Initialize mountain car 
    car = MountainCar(mode, 1) # fixes at pos = 0.8, vel = 1
    # Set up
    Q = []
    num_actions, actions = 3, (0,1,2)
    weights = np.zeros((car.state_space, num_actions))
    bias = 0
    # Do actions
    for episode in range(episodes):
        num_iters = 0
        while num_iters < max_iterations:
            num_iters += 1
            # With prob epsilon, pick random action
            prob = random.random() 
            action = random.choice(action) if (prob < epsilon) else  0 #getBestAction(Q, state, actions)
            # Observe sample 
            (state, reward, done) = car.step(action)
            if done:
                print('done')
                car.reset()
            else:
                # As array
                state = np.fromiter(state.values(), dtype=float)
                # Update q 
                Q[state][action] = np.dot(state, w[state][action]) + bias
                sample = reward + gamma * (maxQnext)
                gradient = state
                # update weights 
                w[state][action] = w[state][action] - (alpha * (Q[state][action] - sample)) * gradient
                bias = bias - (alpha * (Q[state][action] - sample)) * gradient
        car.reset()
    car.close()

def update_Q(Q, state, weights):
    if state not in Q:
        Q[state] = []



# def update_weight(state, reward, action, weights, bias, learning_rate, gamma):
#     sample = reward + gamma(maxQnext)
#     currQ = np.dot(state, weights[:,action]) + bias
#     gradient = state
#     weights = weights - learning_rate*(currQ - sample)*gradient
#     bias = bias - learning_rate*(currQ - sample)*gradient 
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
