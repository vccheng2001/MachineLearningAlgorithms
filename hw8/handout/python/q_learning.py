import sys
from environment import MountainCar
import numpy as np
import random 
import math 

def main():
    (program, mode, weight_out, returns_out, episodes, max_iterations, epsilon, gamma, alpha) = sys.argv
    epsilon, gamma, alpha, episodes, max_iterations = float(epsilon), float(gamma), float(alpha), int(episodes), int(max_iterations)
    # Set up
    actions, num_actions = (0,1,2), 3 
    car = MountainCar(mode=mode)
    state = tuple(car.state)
    ss = car.state_space
    Q = {}
    # weights: 2 by 3 matrix 
    w = np.zeros((ss, num_actions))
    bias = 0
    # Output files 
    w_out = open(weight_out, 'w')
    r_out = open(returns_out, 'w')
    # Do actions
    for episode in range(episodes):
        num_iters = 0
        total_rewards = 0
        while num_iters < max_iterations:
            num_iters += 1
            # Init Q[state] for all actions
            Q = init_Q(Q, state, actions)
            # With prob epsilon, pick random action
            prob = random.random() 
            action = random.choice(actions) if (prob < epsilon) else getBestAction(Q, state, actions)
            # Observe sample 
            (next_state, reward, done) = car.step(action)
            next_state = tuple(next_state)
            # Add curr reward 
            total_rewards += reward 
            # Init next state Q
            Q = init_Q(Q, next_state, actions)
            # Sample
            sample = reward + gamma * bestQVal(Q, next_state) # get best value
            # Calculate q
            Q[state][action] = np.dot(np.asarray(state), w[:,action]) + bias
            # Update weights
            w_gradient = np.asarray(state).T
            diff = Q[state][action] - sample
            w[:,action] = w[:,action] - alpha*diff*w_gradient
            bias =  bias - alpha*diff*1
            state = next_state
            if done:
                car.reset()
                break
        # Print rewards 
        r_out.write(str(total_rewards)+ "\n")
        car.reset()
    # Weight outputs 
    w_out.write(str(bias)+'\n')
    w_list = w.flatten(order='C')
    for i in w_list:
        w_out.write(str(i) + '\n')
    # Close
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
