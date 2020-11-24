import sys
from environment import MountainCar
import pyglet
import numpy as np
import random 
import math 

def main():
    (program, mode, weight_out, returns_out, episodes, max_iterations, epsilon, gamma, alpha) = sys.argv
    epsilon, gamma, alpha, episodes, max_iterations = float(epsilon), float(gamma), float(alpha), int(episodes), int(max_iterations)
    # Initialize mountain car 
    car = MountainCar(mode, None) # fixes at pos = 0.8, vel = 1
    # Set up
    Q, w, bias = {}, {}, 0
    num_actions, actions = 3, (0,1,2)
    ss = car.state_space
    # Do actions
    for episode in range(episodes):
        num_iters = 0
        while num_iters < max_iterations:
            num_iters += 1
            print('curr state', car.state)
            state = car.state 
            state = tuple(state)
            # Init Q[state] for all actions
            Q = init_Q(Q, state, actions)
            # With prob epsilon, pick random action
            prob = random.random() 
            action = random.choice(actions) if (prob < epsilon) else getBestAction(Q, state, actions)
            print('prob', prob)
            print('action', action)
            # Observe sample 
            (next_state, reward, done) = car.step(action)
            next_state = tuple(next_state)
            # Init next_Q 
            Q = init_Q(Q, next_state, actions)
            print('next state, reward, done', next_state, reward, done)
            # Sample
            sample = reward + gamma * bestQVal(Q, next_state) # get best value
            # print(reward, gamma, bestQVal(Q, next_state), sample)
            w = update_w(Q, state, actions, w, ss)
            Q[state][action] = np.dot(np.asarray(state), np.asarray(w[state][action])) + bias
            # print(np.asarray(state), np.asarray(w[state][action]), bias, Q[state][action])
            wgradient = state
            # update weights
            diff = Q[state][action] - sample
            # print(sample, Q[state][action], diff)
            w[state][action] = w[state][action] - np.multiply(alpha * diff,  wgradient)
            bias =  bias - (alpha * diff * 1)
            print('weight', w[state])
            print('bias', bias)
            if done: car.reset()
        car.reset()
    car.close()


def init_Q(Q, state, actions):
    if not state in Q:
        Q[state] = {}
    for action in actions:
        if not action in Q[state]:
            Q[state][action] = 0
    return Q 

def bestQVal(Q, next_state):
    return max(Q[next_state].values())

def update_w(Q, state, actions, w, ss):
    if not state in w:
        w[state] = {}
    for action in actions:
        if not action in w[state]:
            w[state][action] = np.zeros(ss)
    return w


# Return best action for given state 
def getBestAction(Q, state, actions):
    bestAction = None
    bestQ= float('-inf')
    # Loop through actions
    for action in actions:
        # Get Q val of state, action
        currQ = Q[state][action]
        if currQ > bestQ:
            bestAction = action
            bestQ = currQ
    print("BESTQ", bestQ)
    return bestAction 

if __name__ == "__main__":
    main()
