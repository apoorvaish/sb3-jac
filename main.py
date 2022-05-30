# %%
import gym
import math
import numpy as np
import torch as T
from stable_baselines3 import SAC
from sim.reacher import ReacherEnv
from stable_baselines3.common.callbacks import BaseCallback


def calc_jac_from_obs(obs, eps, l, kp=0.003):
    th1 = obs[0][0]
    th2 = obs[0][1]
    l1 = eps*l
    l2 = (1-eps)*l 
    j11 = - l1*math.sin(th1) - l2*math.sin(th1+th2)
    j12 = -l2*math.sin(th1+th2)
    j21 = l1*math.cos(th1) + l2*math.cos(th1+th2)
    j22 = l2*math.cos(th1+th2)
    jac = np.array([[j11, j12], [j21, j22]])
    jac_inv = np.linalg.inv(jac) * kp
    
    return jac_inv

def custom_predict(observation: T.Tensor, deterministic: bool = False) -> T.Tensor:
    action = model.actor(observation, deterministic)
    # jac_inv = calc_jac_from_obs(observation, 0.5, 0.2)
    # action = T.matmul(action, T.tensor(jac_inv.astype(np.float32)))
    # action = T.tanh(action)
    action = T.clamp(action, -1, 1)
    return action


env = ReacherEnv()
model = SAC('MlpPolicy', env, tensorboard_log="./logs/", verbose=1)
model.policy._predict = custom_predict
model.learn(total_timesteps=4e4)
# %%
model.learn(total_timesteps=2e4)

# %%
import imageio

env = ReacherEnv()
obs = env.reset()
done = False
total_reward = 0
images = []
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    img = env.render(mode='rgb_array')
    images.append(img)

print(total_reward)
imageio.mimsave('SAC.gif', [np.array(img)
                for i, img in enumerate(images) if i % 2 == 0])

# %%
