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
        print("Episode number %d" % i)
        done = False
        num_iters = 0
        total_rewards = 0
        # Convert dict to tuple with dict's values 
        state_dict = car.reset()
        state = tuple(state_dict.values())
        while True:
            print("\n ITER NUMBER %d \n" % num_iters)
            num_iters += 1

            if num_iters > max_iterations:
                car.reset()
                break
        
            if not state in Q:
                # initialize Q values for each action to 0 
                Q[state] = {0:0, 1:0, 2:0}
            print("Q is...", Q)

            # With prob epsilon, pick random action
            prob = random.random() 
            print("Prob is...", prob)
            action = random.choice(actions) if (prob < epsilon) else getBestAction(Q, state, actions)
            
            # Observe sample 
            (next_state_dict, reward, done) = car.step(action)
            print("Nextstate, Reward, Done", next_state_dict, reward, done)
            next_state = tuple(next_state_dict.values())
            print("Nextstate", next_state)
            # Add curr reward 
            total_rewards += reward 
            print("Total rewards", total_rewards)
      
            if done: 
                car.reset() 
                break 
            else: 
                # Sample
                if not next_state in Q: # Init next_state Q values to 0
                    Q[next_state] = {0:0, 1:0, 2:0}
                print("Q with ns", Q)
                print("Gamma", gamma)
                sample = reward + (gamma * bestQVal(Q, next_state)) # get best value
                print("bestQNext", bestQVal(Q, next_state))
                print("Sample", sample)

                print("Taking the dot product of....")
                print("State as array", np.asarray(state))
                print("Wa", w[action]) 
                print("Bias", bias)
                Q[state][action] = np.dot(np.asarray(state), w[action]) + bias
                print("Q[state][action]", Q[state][action])

                # Update weights
                w_gradient = np.asarray(state).T
                print("Weight gradient", w_gradient)
                diff = Q[state][action] - sample
                print("Diff=Q[s][a] - sample", diff)
                print("Wa", w[action])
                print("Alpha*diff*w_gradient", alpha*diff*w_gradient)
                w[action] = w[action] - alpha*diff*w_gradient
                print("Updated Wa", w[action])
                print("Updated W", w)
                bias = bias - alpha*diff*1
                print("updated bias", bias)
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


def bestQVal(Q, next_state):
    print("Q nextstate", Q[next_state])
    return max(Q[next_state].values())

# Return best action for given state 
def getBestAction(Q, state, actions):
    print("Getting best action")
    bestAction = None
    bestQ= float('-inf')
    # Loop through actions
    for action in actions:
        # Get Q val of state, action
        currQ = Q[state][action]
        if currQ > bestQ:
            bestAction = action
            bestQ = currQ
    print("Chose action %d" % bestAction)
    return bestAction 

if __name__ == "__main__":
    main()
