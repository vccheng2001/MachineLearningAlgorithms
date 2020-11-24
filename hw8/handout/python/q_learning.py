import sys
from environment import MountainCar
import pyglet
import numpy as np
import random 
import math 

# def map_state_to_index(state):
#     (pos, vel) = state 
#     col = (pos + 1.2)/0.36
#     row = (vel + 0.07)/0.03 
#     print(math.floor(row), math.floor(col))

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
        state = car.state 
        num_iters = 0
        while num_iters < max_iterations:
            num_iters += 1
            # With prob epsilon, pick random action
            prob = random.random() 
            action = random.choice(actions) if (prob < epsilon) else getBestAction(Q, state, actions)
            # Observe sample 
            (next_state, reward, done) = car.step(action)
            sample = reward + gamma * bestQVal(Q, next_state) # get best value
            w = update_w(Q, state, action, w, bias)
            Q = update_Q(Q, state, action, w, bias)
            gradient = state
            # update weights 
            w[state][action] = w[state][action] - (alpha * (Q[state][action] - sample)) * gradient
            bias = bias - (alpha * (Q[state][action] - sample)) * gradient
            if done: car.reset()
            else: state = next_state
        car.reset()
    car.close()

def bestQVal(Q, next_state):
    if not next_state in Q:
        Q[next_state] = {}
        return 0
    else:
        return max(Q[next_state].values())

def update_w(Q, state, action, w, bias):
    if not state in w:
        w[state] = []
    if not action in w[state]:
        w[state][action] = 0
    else:
        w[state][action] = np.dot(state, w[state][action]) + bias
    return w

def update_Q(Q, state, action, w, bias):
    if not state in Q:
        Q[state] = {}
    Q[state][action] = np.dot(state, w[state][action]) + bias
    return Q 


# def update_weight(state, reward, action, weights, bias, learning_rate, gamma):
#     sample = reward + gamma(maxQnext)
#     currQ = np.dot(state, weights[:,action]) + bias
#     gradient = state
#     weights = weights - learning_rate*(currQ - sample)*gradient
#     bias = bias - learning_rate*(currQ - sample)*gradient 
#     return weights 

# Return best action for given state 
def getBestAction(Q, state, actions):
    bestAction = None
    bestQ= float('-inf')
    for action in actions:
        currQ = Q[state][action] 
        if currQ >= bestQ:
            bestAction = action
            bestQ = currQ
    return bestAction 

if __name__ == "__main__":
    main()
