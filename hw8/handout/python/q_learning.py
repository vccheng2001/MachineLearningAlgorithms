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
    epsilon, gamma, alpha, episodes, max_iterations = float(epsilon), float(gamma), float(alpha), int(episodes), int(max_iterations)
    # Initialize mountain car 
    car = MountainCar(mode, None) # fixes at pos = 0.8, vel = 1
    # Set up
    Q = {}
    num_actions, actions = 3, (0,1,2)
    w = {}
    bias = 0
    ss = car.state_space
    # Do actions
    for episode in range(episodes):
        num_iters = 0
        while num_iters < max_iterations:
            num_iters += 1
            print('curr state', car.state)
            state = car.state 
            state = tuple(state)
            # With prob epsilon, pick random action
            prob = random.random() 
            action = random.choice(actions) if (prob < epsilon) else getBestAction(Q, state, actions)
            print('action', action)
            # Observe sample 
            (next_state, reward, done) = car.step(action)
            next_state = tuple(next_state)
            print('next state', next_state)
            # Sample
            sample = reward + gamma * bestQVal(Q, next_state) # get best value
            w = update_w(Q, state, action, w, bias, ss)
            Q = update_Q(Q, state, action, w, bias)
            gradient = state
            # update weights
            diff = Q[state][action] - sample
            w[state][action] = w[state][action] - np.multiply(alpha * diff,  gradient)
            bias =  bias - (alpha * diff * 1)
            print('weight', w[state])
            print('bias', bias)
            if done: car.reset()
            else: state = next_state
        car.reset()
    car.close()

def bestQVal(Q, next_state):
    if not next_state in Q:
        Q[next_state] = {}
        return 0
    elif Q[next_state] == {}:
        return 0
    else:
        return max(Q[next_state].values())

def update_w(Q, state, action, w, bias, ss):
    if not state in w:
        w[state] = {}
    if not action in w[state]:
        w[state][action] = np.zeros(ss)
    return w

def update_Q(Q, state, action, w, bias):
    if not state in Q:
        Q[state] = {}
    # print('-----')
    # print(state)
    # print(np.asarray(state))
    # print(w[state][action])
    # print(np.asarray(w[state][action]))
    Q[state][action] = np.dot(np.asarray(state), np.asarray(w[state][action])) + bias
    print('UPDATE Q Q[state][action]', Q[state][action])
    return Q 

# Return best action for given state 
def getBestAction(Q, state, actions):
    if not state in Q:
        Q[state] = {}
    bestAction = None
    bestQ= float('-inf')
    for action in actions:
        # If not initialized, set to 0
        if not action in Q[state]:
            Q[state][action] = 0
        # Get Q val of state, action
        currQ = Q[state][action]
        if currQ > bestQ:
            bestAction = action
            bestQ = currQ
    print("BESTQ", bestQ)
    return bestAction 

if __name__ == "__main__":
    main()
