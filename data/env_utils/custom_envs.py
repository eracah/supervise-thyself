from gym.envs.registration import register
import gym
from gym import spaces
from ple import PLE
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
class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
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
    


class FlappyBirdWrapper(InfoWrapper):
    y_coord = env.env.game_state.game.player.pos_y
    y_coord = bucket_coord(y_coord,env.num_buckets,env.env.game_state.game.height)
    max_pipe_dist = 305
    x_pipes = [env.env.game_state.game.pipe_group.sprites()[i].x for i in range(3)]
    min_x = min(x_pipes)
    ind_x = x_pipes.index(min_x)
    if min_x < 0:
        ind_x = (ind_x + 1) % 3
    assert ind_x < 3, "whoa whoa %i, %i, %i" %(x_pipes[0],x_pipes[1], x_pipes[2])
    x_pipe = x_pipes[ind_x]
    if x_pipe > max_pipe_dist:
        x_pipe = 0

        
    pipe_x_coord = bucket_coord(x_pipe, env.num_buckets,max_pipe_dist)
    latent_dict = dict(y_coord=y_coord, pipe_x_coord=pipe_x_coord)
class PLEEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, game_name, display_screen=True):
        # set headless mode
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        # open up a game state to communicate with emulator
        import importlib
        game_module_name = ('ple.games.%s' % game_name).lower()
        game_module = importlib.import_module(game_module_name)
        game = getattr(game_module, game_name)()
        self.game_state = PLE(game, fps=30, frame_skip=2, display_screen=display_screen)
        self.game_state.init()
        self._action_set = self.game_state.getActionSet()
        self.action_space = spaces.Discrete(len(self._action_set))
        self.screen_width, self.screen_height = self.game_state.getScreenDims()
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_width, self.screen_height, 3))
        self.viewer = None
        self.count = 0

    def _step(self, a):
        reward = self.game_state.act(self._action_set[a])
        state = self._get_image()
        #import scipy.misc
        #scipy.misc.imsave('outfile'+str(self.count)+'.jpg', state)
        #self.count = self.count+1
        terminal = self.game_state.game_over()
        #print(randomAction)
        #print(a,self._action_set[a])
        return state, reward, terminal, {}

    def _get_image(self):
        #image_rotated = self.game_state.getScreenRGB()
        image_rotated = np.fliplr(np.rot90(self.game_state.getScreenRGB(),3)) # Hack to fix the rotated image returned by ple
        return image_rotated

    @property
    def _n_actions(self):
        return len(self._action_set)

    # return: (states, observations)
    def _reset(self):
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_width, self.screen_height, 3))
        self.game_state.reset_game()
        state = self._get_image()
        return state

    def _render(self, mode='human', close=False):
        #print('HERE')
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = self._get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)


    def _seed(self, seed):
        rng = np.random.RandomState(seed)
        self.game_state.rng = rng
        self.game_state.game.rng = self.game_state.rng

        self.game_state.init()
        

from gym.envs.registration import registry, register, make, spec
#from gym_ple.ple_env import PLEEnv
# Pygame
# ----------------------------------------
for game in ['originalGame','nosemantics','noobject','nosimilarity','noaffordance']: 
    #,'Catcher',  'FlappyBird','Pixelcopter', 'PuckWorld','RaycastMaze', 'Snake', 'WaterWorld']:
    nondeterministic = False
    register(
        id='{}-v0'.format(game),
        entry_point='data.env_utils.custom_envs:PLEEnv',
        kwargs={'game_name': game, 'display_screen':False},
        tags={'wrapper_config.TimeLimit.max_episode_steps': 100000},
        nondeterministic=nondeterministic,
    )        
