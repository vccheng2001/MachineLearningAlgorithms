import sys
from environment import MountainCar
import pyglet
import numpy as np
import random 
import math 

def main():
    (program, mode, weight_out, returns_out, episodes, max_iterations, epsilon, gamma, alpha) = sys.argv
    epsilon, gamma, alpha, episodes, max_iterations = float(epsilon), float(gamma), float(alpha), int(episodes), int(max_iterations)

    w_out = open(weight_out, 'w')
    r_out = open(returns_out, 'w')
    # Do actions
    for episode in range(episodes):
        print('episode', episode)
        Q, w, bias = {}, {}, 0
        # Initialize mountain car
        car = MountainCar(mode, 1)
        ss = car.state_space
        num_iters, total_rewards = 0, 0
        actions, num_actions = (0,1,2), 3
        while num_iters < max_iterations:
            num_iters += 1
            state = car.state 
            state = tuple(state)
            # Init Q[state] for all actions
            Q = init_Q(Q, state, actions)
            # With prob epsilon, pick random action
            prob = random.random() 
            action = random.choice(actions) if (prob < epsilon) else getBestAction(Q, state, actions)
            # Observe sample 
            (next_state, reward, done) = car.step(action)
            # Add curr reward 
            total_rewards += reward 
            if done: 
                car.reset()
                break
            else:
                next_state = tuple(next_state)
                # Init next state Q
                Q = init_Q(Q, next_state, actions)
                # Sample
                sample = reward + gamma * bestQVal(Q, next_state) # get best value
                w = update_w(Q, state, actions, w, ss)
                # Calculate q
                Q[state][action] = np.dot(np.asarray(state), np.asarray(w[state][action])) + bias
                # Update weights
                wgradient = state
                diff = Q[state][action] - sample
                w[state][action] = w[state][action] - np.multiply(alpha * diff,  wgradient)
                bias =  bias - (alpha * diff * 1)
        # Print
        r_out.write(str(total_rewards)+ "\n")
        w_out.write(str(bias)+'\n')
        for state in w:
            w_out.write(str(w[state][0][0])+'\n')
            w_out.write(str(w[state][0][1])+'\n')
            w_out.write(str(w[state][1][0])+'\n')
            w_out.write(str(w[state][1][1])+'\n')
            w_out.write(str(w[state][2][0])+'\n')
            w_out.write(str(w[state][2][1])+'\n')

        car.reset()
    car.close()
    w_out.close()
    r_out.close()


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
    return bestAction 

if __name__ == "__main__":
    main()
