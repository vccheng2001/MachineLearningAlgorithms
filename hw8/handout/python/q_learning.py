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
    car = MountainCar(mode=mode,fixed=1)
    actions, num_actions = (0,1,2), 3
    # Weights: 2 by 3 matrix 
    Q = {}
    w = np.zeros((2,num_actions))
    bias = 0


    # Do actions
    for i in range(episodes):
        # Initialize 
        num_iters = 0
        total_rewards = 0
        # Raw dictionary 
        state_dict = car.reset()
        # Convert to numpy array 
        state = np.asarray(list(state_dict.values()))

        while num_iters < max_iterations:   
            num_iters += 1

            # E greedy 
            action = getAction(Q, state, actions, epsilon, w, bias)
            
            # Observe sample 
            (next_state_dict, reward, done) = car.step(action)

            # Add current reward 
            total_rewards += reward 

            # Next state
            next_state = np.asarray(list(next_state_dict.values()))
            next_action = getBestAction(Q, next_state, actions, w, bias)
            next_state_max_Q = QValue(next_state, next_action, w, bias)
    
            # Sample 
            sample = reward + (gamma * next_state_max_Q)
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
def getAction(Q, state, actions, epsilon, w, bias):
    prob = random.random() 
    if (prob < epsilon):
        return random.choice(actions)
    else:
        return getBestAction(Q, state, actions, w, bias)                

# Returns Q value 
def QValue(state, action, w, bias):
    return np.dot(state, w[:,action]) + bias

# Return best action for given state 
def getBestAction(Q, state, actions, w, bias):
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
