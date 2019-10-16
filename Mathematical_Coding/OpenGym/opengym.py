# opengym.py
"""Volume 2: Open Gym
<Mark Rose>
<Section 2>
<9/8/19>
"""

import gym
import numpy as np
from IPython.display import clear_output
import random
import time

def find_qvalues(env,alpha=.1,gamma=.6,epsilon=.1):
    """
    Use the Q-learning algorithm to find qvalues.

    Parameters:
        env (str): environment name
        alpha (float): learning rate
        gamma (float): discount factor
        epsilon (float): maximum value

    Returns:
        q_table (ndarray nxm)
    """
    #Make an environment and q-table
    env = gym.make(env)
    q_table = np.zeros((env.observation_space.n,env.action_space.n))

    # We train this here
    for i in range(1,100001):
        # Reset state
        state = env.reset()

        epochs, penalties, reward, = 0,0,0
        done = False

        while not done:
            # Accept based on alpha
            if random.uniform(0,1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            # Take action
            next_state, reward, done, info = env.step(action)

            # Calculate new qvalue
            old_value = q_table[state,action]
            next_max = np.max(q_table[next_state])

            new_value = (1-alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            #Check for a penalty
            if reward == -10:
                penalties += 1

            #Get the next state
            state = next_state
            epochs += 1

        #Print out the episode number
        if i % 100 == 0:
            clear_output(wait=True)
            print(f"Episode: {i}")

    print("Training finished.")
    return q_table

# Problem 1
def random_blackjack(n):
    """
    Play a random game of Blackjack. Determine the
    percentage the player wins out of n times.

    Parameters:
        n (int): number of iterations

    Returns:
        percent (float): percentage that the player
                         wins
    """
    #Initialize the enviroment and the wins
    env = gym.make("Blackjack-v0")
    wins = 0

    #Reset the environment n times
    for i in range(n):
        env.reset()
        while True:
            #take random actions until game is over
            draw = env.step(env.action_space.sample())
            if draw[2] == True:
                #Now count the overwall wins
                if draw[1] == 1:
                    wins += 1
                break

    env.close()
    return wins/n


# Problem 2
def blackjack(n=11):
    """
    Play blackjack with naive algorithm.

    Parameters:
        n (int): maximum accepted player hand

    Return:
        reward (int): total reward
    """
    #Initialize the enviroment, rewards, and number of games
    env = gym.make("Blackjack-v0")
    rewards = 0
    num_games = 10000

    #reset the enviroment for each game
    for i in range(num_games):
        obs = env.reset()
        while True:
            #draw another card if hand is less than or equal to end
            if obs[0] <= n:
                action = 1
            #end game if hand is more than n
            else:
                action = 0
            #get the reward for the game
            obs, reward, done, info = env.step(action)
            if action == 0:
                rewards += reward
                break

    env.close()
    return rewards/num_games

# Problem 3
def cartpole():
    """
    Solve CartPole-v0 by checking the velocity
    of the tip of the pole

    Return:
        time (float): time cartpole is held vertical
    """

    #Initialize the environment and time
    env = gym.make("CartPole-v0")
    start = time.time()

    #take actions and visulalize each actions
    try:
        obs=env.reset()
        done = False
        obs, reward, done, info = env.step(env.action_space.sample())
        while not done:
            #take actions based on the velocity of the pole
            env.render()
            if obs[3] < -.1:
                action = 0
            if obs[3] > .1:
                action = 1
            obs, reward, done, info = env.step(action)
            if done:
                break

    finally:
        #end the time and close environment
        end = time.time()
        env.close()

    return end-start


# Problem 4
def car():
    """
    Solve MountainCar-v0 by checking the position
    of the car.

    Return:
        time (float): time to solve environment
    """
    #Initialize the environment and time
    env = gym.make("MountainCar-v0")
    start = time.time()

    #take actions and visulalize each actions
    try:
        obs=env.reset()
        done = False
        obs, reward, done, info = env.step(env.action_space.sample())
        while not done:
            #Based on the velocity and position of the car take actions
            env.render()
            if obs[0] < 0 and obs[1] < 0:
                action = 0
            if obs[0] < 0 and obs[1] > 0:
                action = 2
            if obs[0] > 0 and obs[1] < 0:
                action = 0
            if obs[0] > 0 and obs[1] > 0:
                action = 2
            obs, reward, done, info = env.step(action)
            if done:
                break

    finally:
        #End the time and close the environment
        end = time.time()
        env.close()

    return end-start


# Problem 5
def taxi(q_table):
    """
    Compare naive and q-learning algorithms.

    Parameters:
        q_table (ndarray nxm): table of qvalues

    Returns:
        naive (int): reward of naive algorithm
        q_reward (int): reward of Q-learning algorithm
    """
    #Initialize the environment rewards
    env = gym.make("Taxi-v2")
    reward1 = 0
    reward2 = 0
    #Run the experiment 10000 times
    for i in range(10000):
        try: 
            #Move around randomly and track the rewards
            env.reset()
            done = False
            while not done:
                obs,reward, done, infor = env.step(env.action_space.sample())
                reward1 +=reward
                
            #Reset the environment for the other experiment
            done = False
            obs = env.reset()
            
            #Move based on the q_table
            while not done:
                obs, reward, done, infor = env.step(np.argmax(q_table[obs]))
                reward2 += reward
                #Once its over record the rewards
                
        finally:
            env.close()
    #Return the average rewards
    return(reward1/10000, reward2/10000)
    raise NotImplementedError("Problem 5 Incomplete")
