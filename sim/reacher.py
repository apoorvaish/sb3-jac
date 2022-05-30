# %%
import numpy as np
import os
from gym import utils
from gym.envs.mujoco import mujoco_env
import xml.etree.ElementTree as ET

class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        self.xml_path = os.getcwd() + '/sim/reacher.xml'
        # self.change_target(self.xml_path)
        self.ep_count = 0
        self.ep_len = 100
        
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, self.xml_path, 2)

    def step(self, a):
        a = np.clip(a, -1, 1)
        self.ep_count += 1
        vec = self.sim.data.geom_xpos[5] - self.sim.data.geom_xpos[4]
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(a).sum()
        reward = reward_dist + reward_ctrl
        if self.ep_count < 5 or ((self.ep_count < self.ep_len) and (abs(reward_dist) > 0.01)):
            self.do_simulation(a, self.frame_skip)
            obs = self._get_obs()
            return obs, reward, False, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
        else:
            self.close()
            return -1, 0, True, dict(reward_dist=0, reward_ctrl=0)

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent
        # look at center of the sim
        v.cam.lookat[:] = self.model.stat.center[:]
        v.cam.elevation = -90

    def change_target(self, xml_path, x=0, y=0):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        if x == 0 and y == 0:
            while True:
                target = np.random.uniform(low=-0.2, high=0.2, size=2)
                if np.linalg.norm(target) < 0.2:
                    break  
            x = target[0]
            y = target[1]
        else:
            target = [x, y]
        pos = str(round(target[0], 3)) + ' ' + str(round(target[1], 3)) + ' 0'
        root.find('worldbody')[-1].find('geom').attrib['pos'] = pos
        tree.write(xml_path)
        return x, y
    
    def reset_model(self):
        # self.change_target(self.xml_path)
        self.ep_count = 0
        qpos = (
            self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
            + self.init_qpos
        )
        while True:
            self.goal = self.np_random.uniform(low=0.1, high=0.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate(
            [
                np.cos(theta), 
                np.sin(theta), 
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat[:2], 
                self.get_body_com("fingertip") - self.get_body_com("target"),
                # theta,
                # self.sim.data.geom_xpos[4][:2],
                # self.sim.data.geom_xpos[5][:2] # Target position
                # self.sim.data.geom_xpos[5][:2] - self.sim.data.geom_xpos[4][:2],   
            ]
        )

# %%
