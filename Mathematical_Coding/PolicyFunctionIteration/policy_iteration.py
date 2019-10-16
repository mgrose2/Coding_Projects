# policy_iteration.py
"""Volume 2: Policy Function Iteration.
<Mark Rose>
<Section 2>
<4/16/19>
"""

import numpy as np
import scipy.linalg as la
import gym
from gym import wrappers


# Intialize P for test example
#Left =0
#Down = 1
#Right = 2
#Up= 3
P = {s : {a: [] for a in range(4)} for s in range(4)}
P[0][0] = [(0, 0, 0, False)]
P[0][1] = [(1, 2, -1, False)]
P[0][2] = [(1, 1, 0, False)]
P[0][3] = [(0, 0, 0, False)]
P[1][0] = [(1, 0, -1, False)]
P[1][1] = [(1, 3, 1, True)]
P[1][2] = [(0, 0, 0, False)]
P[1][3] = [(0, 0, 0, False)]
P[2][0] = [(0, 0, 0, False)]
P[2][1] = [(0, 0, 0, False)]
P[2][2] = [(1, 3, 1, True)]
P[2][3] = [(1, 0, 0, False)]
P[3][0] = [(0, 0, 0, True)]
P[3][1] = [(0, 0, 0, True)]
P[3][2] = [(0, 0, 0, True)]
P[3][3] = [(0, 0, 0, True)]

# Problem 1
def value_iteration(P, nS ,nA, beta = 1, tol=1e-8, maxiter=3000):
    """Perform Value Iteration according to the Bellman optimality principle.

    Parameters:
        P (dict): The Markov relationship 
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.
        maxiter (int): The maximum number of iterations.

    Returns:
       v (ndarray): The discrete values for the true value function.
    """
    #Initiate vectors for the states
    v_states = []
    v_old = np.zeros(nS)
    v_new = np.zeros(nS)
    current_state = 0
    current_action = 0
    #Iterate a set amount of times
    for i in range(1,maxiter+1):
        for s in range(nS):
            max_a = []
            for a in range(nA):
                my_val = 0
                #Use the formula found in 18.3 to get a max value for V
                for q in range(len(P[s][a])):
                    info = P[s][a][q]
                    my_val+=(info[0]*(info[2]+beta*v_old[info[1]]))
                max_a.append(my_val)
            v_new[s] = max(max_a)
        #Break if the value function has converged
        if la.norm(v_new-v_old) < tol:
            break
        v_old = v_new.copy()
    return(v_new,i)
            
    raise NotImplementedError("Problem 1 Incomplete")

# Problem 2
def extract_policy(P, nS, nA, v, beta = 1.0):
    """Returns the optimal policy vector for value function v

    Parameters:
        P (dict): The Markov relationship 
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        v (ndarray): The value function values.
        beta (float): The discount rate (between 0 and 1).

    Returns:
        policy (ndarray): which direction to move in from each square.
    """
    #Initiate a policy
    pol = np.zeros(nS)
    for s in range(nS):
        argmax_a = []
        for a in range(nA):
            my_val = 0
            #Use the formula found in 18.6 to get a new policy
            for q in range(len(P[s][a])):
                info = P[s][a][q]
                my_val+=(info[0]*(info[2]+beta*v[info[1]]))
            #Append the action
            argmax_a.append(my_val)
        pol[s] = np.argmax(argmax_a)
    return(pol)
    raise NotImplementedError("Problem 2 Incomplete")
    
# Problem 3
def compute_policy_v(P, nS, nA, policy, beta=1.0, tol=1e-8):
    """Computes the value function for a policy.
    
    Parameters:
        P (dict): The Markov relationship 
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        policy (ndarray): The policy to estimate the value function.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.
    
    Returns:
        v (ndarray): The discrete values for the true value function.
    """
    #Initiate a v matrix
    v_old = np.zeros(nS)
    v_new = np.ones(nS)
    #While the value function has not converged calculate the following
    while(la.norm(v_new-v_old)>=tol):
        v_old = v_new.copy()
        v_new = np.zeros(nS)
        #Use the equation found in 18.7
        for s in range(nS):
            for q in range(len(P[s][policy[s]])):
                info = P[s][policy[s]][q]
                v_new[s] += info[0]*(info[2]+beta*v_old[info[1]])
    #Return the new value function
    return(v_new)
    raise NotImplementedError("Problem 3 Incomplete")

# Problem 4
def policy_iteration(P, nS, nA, beta=1, tol=1e-8, maxiter=200):
    """Perform Policy Iteration according to the Bellman optimality principle.

    Parameters:
        P (dict): The Markov relationship 
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.
        maxiter (int): The maximum number of iterations.

    Returns:
        policy (ndarray): which direction to moved in each square.
    """
    #Get a simple policy
    old_pol = np.zeros(nS)
    for i in range(1,maxiter+1):
        #Get a value function from the policy and update
        vk = compute_policy_v(P,nS,nA,old_pol,beta,tol)
        new_pol = extract_policy(P,nS,nA,vk,beta)
        #If the policy has converged break
        if(la.norm(new_pol-old_pol) < tol):
            break
        old_pol = new_pol.copy()
    #Return the value function, policy, and the number of iterations
    return(vk, new_pol,i)
    raise NotImplementedError("Problem 4 Incomplete")
    
# Problem 5
def frozenlake(basic_case=True, M=1000):
    """ Finds the optimal policy to solve the FrozenLake problem.
    
    Parameters:
    basic_case (boolean): True for 4x4 and False for 8x8 environemtns. 
    M (int): The number of times to run the simulation.
    
    Returns:
    vi_policy (ndarray): The optimal policy for value iteration.
    vi_total_rewards (float): The average expected value for following the value iteration optimal policy.
    pi_value_func (ndarray): The maximum value functiono for the optimal policy from policy iteration.
    pi_policy (ndarray): The optimal policy for policy iteration.
    pi_total_rewards (float): The average expected value for following the policy iteration optimal policy.
    """
    #Load the environment based on which one we want
    if basic_case ==True:
        env_name = 'FrozenLake-v0'
    else:
        env_name = 'FrozenLake8x8-v0'
    env = gym.make(env_name).env  
    #Get the number of states and actions, and the probabilities
    nS = env.nS
    nA = env.nA
    P = env.P
    reward_val = []
    #Set beta to be some value from 0 to 1
    beta = 1
    #Get a policy from value iteration and run the environment M times
    valit_pol = extract_policy(P,nS,nA,value_iteration(P,nS,nA)[0])
    for i in range(M):
        reward_val.append(run_simulation(env, valit_pol, beta))
    vi_mean = np.mean(reward_val)
    reward_pol = []
    #Get a policy from policy iteration and run the environment M times
    polit_val, polit_pol, num_it = policy_iteration(P,nS,nA)
    for i in range(M):
        reward_pol.append(run_simulation(env, polit_pol, beta))
    pi_mean = np.mean(reward_pol)
    #return the policy's and their respected mean total reward
    return(valit_pol, vi_mean, polit_val, polit_pol, pi_mean)
    raise NotImplementedError("Problem 5 Incomplete")
    
# Problem 6
def run_simulation(env, policy, beta = 1.0):
    """ Evaluates policy by using it to run a simulation and calculate the reward.
    
    Parameters:
    env (gym environment): The gym environment. 
    policy (ndarray): The policy used to simulate.
    beta float): The discount factor.
    
    Returns:
    total reward (float): Value of the total reward recieved under policy.
    """
    #initiate the rewards and let the simulation begin
    done = False
    total_reward = 0
    obs = env.reset()
    k=1
    #If the environment returns done, break the loop
    while(done!=True):
        obs, reward, done, _ = env.step(int(policy[obs]))
        #Get the total reward by adding the current reward and multiplying it by B**k
        total_reward+=reward*(beta**k)
        k+=1
    #Return the total reward
    return(total_reward)
   
    raise NotImplementedError("Problem 6 Incomplete")

