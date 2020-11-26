import sys
from environment import MountainCar
import numpy as np
import random 
import math 

def main():
    (program, mode, weight_out, returns_out, episodes, max_iterations, epsilon, gamma, alpha) = sys.argv
    epsilon, gamma, alpha, episodes, max_iterations = float(epsilon), float(gamma), float(alpha), int(episodes), int(max_iterations)
    # Output files 
    w_out = open(weight_out, 'w')
    r_out = open(returns_out, 'w')
    # Initialize Mountain Car 
    car = MountainCar(mode=mode)
    actions, num_actions = (0,1,2), 3
    # Weights: <dim(S)> by <num_actions> matrix 
    w = np.zeros((car.state_space ,num_actions))
    bias = 0

    # Represent state as numpy array 
    def state_rep(state_dict, mode):
        if mode == "raw":
            state = np.asarray(list(state_dict.values()))
        elif mode == "tile":
            state = np.zeros(2048)
            for key in state_dict:
                state[key] = 1
        return state 

    # Do actions
    for i in range(episodes):
        # Initialize 
        num_iters = 0
        total_rewards = 0
        # Raw dictionary 
        state_dict = car.reset()
        # Convert to numpy array 
        state = state_rep(state_dict, mode)
    
        while num_iters < max_iterations:   
            num_iters += 1

            # E greedy 
            action = getAction(state, actions, epsilon, w, bias)
            
            # Observe sample 
            (next_state_dict, reward, done) = car.step(action)
            
            # Add current reward 
            total_rewards += reward 

            # Next state, get best action for next state 
            next_state = state_rep(next_state_dict, mode)
            next_best_action = getBestAction(next_state, actions, w, bias)
            next_state_best_Q = QValue(next_state, next_best_action, w, bias)
    
            # Sample 
            sample = reward + (gamma * next_state_best_Q)
            diff = QValue(state, action, w, bias) - sample

            # Update weights 
            w[:,action] = w[:,action] - alpha * diff * state
            bias = bias - alpha*diff*1

            # Break if done 
            if not done:
                state = next_state
            else:
                break 

        # Print rewards 
        r_out.write(str(total_rewards)+ "\n")
        
    # Print weight outputs 
    w_out.write(str(bias)+'\n')
    for row in w:
        for elem in row:
            w_out.write(str(elem) + '\n')

    # Close
    car.close()
    w_out.close()
    r_out.close()

# Return action based on epsilon greedy 
def getAction(state, actions, epsilon, w, bias):
    prob = random.random() 
    if (prob < epsilon):
        return random.choice(actions)
    else:
        return getBestAction(state, actions, w, bias)                

# Returns Q value 
def QValue(state, action, w, bias):
    return np.dot(state, w[:,action]) + bias

# Return best action for given state 
def getBestAction(state, actions, w, bias):
    bestAction = None
    bestQ= float('-inf')
    # Loop through actions
    for action in actions:
        # Get Q value of state, action
        currQ = QValue(state, action, w, bias)
        if currQ > bestQ:
            bestAction = action
            bestQ = currQ
    return bestAction 

if __name__ == "__main__":
    main()
