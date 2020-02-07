import matplotlib.pyplot as plt
import numpy as np
import sys
import time
from gym_unity.envs import UnityEnv


print("Python version:")
print(sys.version)

# check Python version
if (sys.version_info[0] < 3):
    raise Exception("ERROR: ML-Agents Toolkit (v0.3 onwards) requires Python 3")

env_name = "./Unity/3DBall"  # Name of the Unity environment binary to launch
env = UnityEnv(env_name, worker_id=2,multiagent=True)

# Examine environment parameters
print(str(env))

# Reset the environment
initial_observations = env.reset()

if len(env.observation_space.shape) == 1:
    # Examine the initial vector observation
    print("Agent observations look like: \n{}".format(initial_observations[0]))
else:
    # Examine the initial visual observation
    print("Agent observations look like:")
    if env.observation_space.shape[2] == 3:
        plt.imshow(initial_observations[0][:,:,:])
    else:
        plt.imshow(initial_observations[0][:,:,0])

for episode in range(10):
    initial_observation = env.reset()
    done = False
    episode_rewards = 0
    while not done:
        actions = [env.action_space.sample() for agent in range(env.number_agents)]
        observations, rewards, dones, info = env.step(actions)
        episode_rewards += np.mean(rewards)
        done = dones[0]

    print("Total reward this episode: {}".format(episode_rewards))


env.close()

