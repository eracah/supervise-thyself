from gym.envs.registration import register
import gym
from gym import spaces
import numpy as np
import os


class InfoWrapper(gym.Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, self.info()

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        info = self.info()
        return observation, info

    def info(self):
        raise NotImplementedError
        
        
class InfoActionWrapper(gym.ActionWrapper):
    
    def step(self, action):
        observation, reward, done, info = super().step(action)
        return observation, reward, done, self.info()

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        info = self.info()
        return observation, info

    def info(self):
        raise NotImplementedError
        
    def action(self):
        raise NotImplementedError
        
    
    

class AtariWrapper(InfoWrapper):
    atari_ram_dict = {"Pitfall-v0":(97,105),
                  "PrivateEye-v0": (63,86)}
    
    def __init__(self, env):
        super().__init__(env)
        self.xind, self.yind = AtariWrapper.atari_ram_dict[self.env.spec.id]
        
           
    def info(self):

        ram = self.env.env.ale.getRAM()
        x,y = ram[self.xind], ram[self.yind]
        return dict(x=x,y=y)     

class LunarLanderWrapper(InfoWrapper):
    def info(self):
        x,y = self.env.env.lander.position
        return dict(x=x,y=y)
        

# I got this from https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py        
class SonicWrapper(InfoActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['DOWN', 'B'], ["RIGHT"]]
        #[['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()
    
    def sonicify_action(self,a):
        return self.action(a)
        
    def info(self):            
        abs_y = self.env.data.lookup_value("y")
        screen_y = self.env.data.lookup_value("screen_y")
        abs_x = self.env.data.lookup_value("x")
        screen_x = self.env.data.lookup_value("screen_x")
        x = abs_x - screen_x
        y = abs_y - screen_y
        
        return dict(x=x, y=y)
        
        
class FlappyBirdWrapper(InfoWrapper):
    
    def info(self):
        y = self.env.env.game_state.game.player.pos_y
        x_pipes = [self.env.env.game_state.game.pipe_group.sprites()[i].x for i in range(3)]
        min_x = min(x_pipes)
        ind_x = x_pipes.index(min_x)
        if min_x < 0:
            ind_x = (ind_x + 1) % 3
        assert ind_x < 3, "whoa whoa %i, %i, %i" %(x_pipes[0],x_pipes[1], x_pipes[2])
        x_pipe = x_pipes[ind_x]
        if x_pipe > max_pipe_dist:
            x_pipe = 0


        return dict(y=y,x_pipe=x_pipe)
    

