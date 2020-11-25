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
    car = MountainCar(mode=mode, fixed=1)
    actions, num_actions = (0,1,2), 3
    # Weights: 2 by 3 matrix 
    Q = {}
    w0 = np.array([0,0])
    w1 = np.array([0,0])
    w2 = np.array([0,0])
    w = [w0, w1, w2]
    bias = 0


    # Do actions
    for i in range(episodes):
        done = False
        num_iters = 0
        total_rewards = 0
        # Convert dict to tuple with dict's values 
        state_dict = car.reset()
        state = tuple(state_dict.values())
        while True:
            num_iters += 1

            if num_iters > max_iterations:
                car.reset()
                break
        
            if not state in Q:
                # initialize Q values for each action to 0 
                Q[state] = {0:0, 1:0, 2:0}

            # With prob epsilon, pick random action
            prob = random.random() 
            action = random.choice(actions) if (prob < epsilon) else getBestAction(Q, state, actions)

            # Observe sample 
            (next_state_dict, reward, done) = car.step(action)
            next_state = tuple(next_state_dict.values())
            # Add curr reward 
            total_rewards += reward 
      
            if done: 
                car.reset() 
                break 
            else: 
                # Sample
                if not next_state in Q: # Init next_state Q values to 0
                    Q[next_state] = {0:0, 1:0, 2:0}
                sample = reward + (gamma * bestQVal(Q, next_state)) # get best value
                Q[state][action] = np.dot(np.asarray(state), w[action]) + bias

                # Update weights
                w_gradient = np.asarray(state).T
                diff = Q[state][action] - sample
                w[action] = w[action] - alpha*diff*w_gradient
                bias = bias - alpha*diff*1
                state = next_state
            
        # Print rewards 
        r_out.write(str(total_rewards)+ "\n")
        
    # Print weight outputs 
    w_out.write(str(bias)+'\n')
    for i in w:
        w_out.write(str(i) + '\n')

    # Close
    car.close()
    w_out.close()
    r_out.close()


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
