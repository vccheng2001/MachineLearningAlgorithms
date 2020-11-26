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
     #   print("Episode number %d" % i)
        done = False
        num_iters = 0
        total_rewards = 0
        # Convert dict to tuple with dict's values 
        state_dict = car.reset()
        state = np.asarray(list(state_dict.values()))

       # print("STARTING STATE FOR EPISODE", state)
        while True:
            #print("\n ITER NUMBER %d \n" % num_iters)
            num_iters += 1
            if num_iters > max_iterations:
                car.reset()
                break

            # E greedy 
            action = getAction(Q, state, actions, epsilon, w, bias)
            
            # Observe sample 
            (next_state_dict, reward, done) = car.step(action)
            next_state = np.asarray(list(next_state_dict.values()))
           
            # Sample
            next_action = getBestAction(Q, next_state, actions, w, bias)
           
            total_rewards += reward 
    

            sample = reward + (gamma * Qvalue(next_state, next_action, w, bias)) # get best value
            diff = Qvalue(state, action, w, bias) - sample


            w[:,action] = w[:,action] - alpha * diff * state
            bias = bias - alpha*diff*1

            if done:
                break 

            state = next_state
        # Print rewards 
        r_out.write(str(total_rewards)+ "\n")
        
    # Print weight outputs 
    w_out.write(str(bias)+'\n')
    for i in w:
        for j in i:
            w_out.write(str(j) + '\n')

    # Close
    car.close()
    w_out.close()
    r_out.close()

def getAction(Q, state, actions, epsilon, w, bias):
    prob = random.random() 
    if (prob < epsilon):
        return random.choice(actions)
    else:
        return getBestAction(Q, state, actions, w, bias)                

def Qvalue(state, action, w, bias):
    return np.dot(state, w[:,action]) + bias

# Return best action for given state 
def getBestAction(Q, state, actions, w, bias):
  #  print("Getting best action")
    bestAction = None
    bestQ= float('-inf')
    # Loop through actions
    for action in actions:
        # Get Q val of state, action
        currQ = Qvalue(state, action, w, bias)
        if currQ > bestQ:
            bestAction = action
            bestQ = currQ
  #  print("Chose action %d" % bestAction)
    return bestAction 

if __name__ == "__main__":
    main()
