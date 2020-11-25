import sys
from environment import MountainCar
import numpy as np
import random 
import math 

def main():
    (program, mode, weight_out, returns_out, episodes, max_iterations, epsilon, gamma, alpha) = sys.argv
    epsilon, gamma, alpha, episodes, max_iterations = float(epsilon), float(gamma), float(alpha), int(episodes), int(max_iterations)
    print('epsilon, gamma')
    # Output files 
    w_out = open(weight_out, 'w')
    r_out = open(returns_out, 'w')
    # Initialize Mountain Car 
    car = MountainCar(mode=mode)
    state = tuple(car.state)
    ss = car.state_space
    actions, num_actions = (0,1,2), 3
    # Weights: 2 by 3 matrix 
    Q = {}
    w = np.zeros((2, num_actions))
    bias = 0


    # Do actions
    for episode in range(1):
        num_iters = 0
        total_rewards = 0
        while num_iters < 3:
            num_iters += 1
            Q = init_Q(Q, state, actions)
            
            # With prob epsilon, pick random action
            prob = random.random() 
            print('prob', prob)
            action = random.choice(actions) if (prob < epsilon) else getBestAction(Q, state, actions)
            print('action chosen', action)

            # Observe sample 
            (next_state, reward, done) = car.step(action)
            next_state = tuple(next_state.values())
            print('state nextstate', state, next_state)

            Q = init_Q(Q, next_state, actions)
            
            # Add curr reward 
            total_rewards += reward 

            # Sample
            sample = reward + (gamma * bestQVal(Q, next_state)) # get best value
            print('reward', reward)
            print('gamma', gamma)
            print('sample', sample)
            Q[state][action] = np.dot(np.asarray(state), w[:,action]) + bias

            print('w', w)
            print('bias', bias)
            print('w[action]', w[:,action])
            print('Q[state][action]', Q[state][action])
            
            # Update weights
            w_gradient = np.asarray(state).T
            diff = Q[state][action] - sample
            print('diff', diff)
            print('alpha', alpha)
            print('alpha * diff * w_gradient', alpha*diff*w_gradient)
            w[:,action] = w[:,action] - alpha*diff*w_gradient
            bias = bias - alpha*diff*1
            print('updated w', w)
            print('updated bias', bias)

            # If done == true, reset 
            if done:
                car.reset()
                break
            else: # else update state 
                state = next_state 
            
        # Reset, print rewards 
        car.reset() 
        r_out.write(str(total_rewards)+ "\n")
        
    # Print weight outputs 
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
    print("NextstateQ", Q[next_state])
    print('bestQNext', max(Q[next_state].values()))
    return max(Q[next_state].values())

# Return best action for given state 
def getBestAction(Q, state, actions):
    print('getting best action')
    print('qstate', Q[state])
    bestAction = None
    bestQ= float('-inf')
    # Loop through actions
    for action in actions:
        # Get Q val of state, action
        currQ = Q[state][action]
        if currQ > bestQ:
            bestAction = action
            bestQ = currQ
    print('bestAction', bestAction)
    return bestAction 

if __name__ == "__main__":
    main()
