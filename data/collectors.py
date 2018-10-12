import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid
import numpy as np
from collections import namedtuple
import copy
from data.utils import setup_env, convert_frame

def get_trans_tuple():
        tuple_fields = ['xs','actions', 'rewards', 'dones', "state_param_dict"]
        

        Transition = namedtuple("Transition",tuple(tuple_fields))
        return Transition

def make_empty_transition():
    transition_constructor = get_trans_tuple()
    num_fields = len(transition_constructor.__dict__["_fields"])
    trans_list = [[] for _ in range(num_fields -1)]
    trans_list.append({})
    trans = transition_constructor(*trans_list)
    return trans
    

class DataCollector(object):
    def __init__(self, policy,
                        env,
                        convert_fxn=convert_frame,
                        frames_per_trans=2):
        self.convert_fxn = convert_fxn
        self.policy = policy
        self.env = env
        self.env.reset()
        #to avoid black frame
        self.env.step(5)
        # datapoints_per_trans=0 means you just collect the current state and don't take an action
        # and get a new state
        assert frames_per_trans >=2, "must have at least an s,a,s triplet"
        self.frames_per_trans = frames_per_trans

        
    def _collect_datapoint(self,obs):
        x = self.convert_fxn(obs)
#         x_coord, y_coord, direction = int(self.env.agent_pos[0]), int(self.env.agent_pos[1]), self.env.agent_dir
#         latent_dict = dict(x_coord=x_coord, y_coord=y_coord, direction=direction)
        latent_dict = self.env.get_latent_dict(self.env)
        return x, latent_dict

    def append_to_trans_ard(self,trans,action,reward, done):
        trans.actions.append(copy.deepcopy(action))
        trans.rewards.append(copy.deepcopy(reward))
        trans.dones.append(copy.deepcopy(done))
        
    def append_to_trans_state(self,trans,x, latent_dict):
        trans.xs.append(copy.deepcopy(x))
        for k,v in latent_dict.items():
            if k not in trans.state_param_dict:
                trans.state_param_dict[k] = [copy.deepcopy(v)]
            else:
                trans.state_param_dict[k].append(copy.deepcopy(v))

        
        
    def collect_transition_per_the_policy(self):
        trans = make_empty_transition()
        
        #obs, reward, done, _ = self.env.step(env.action_space.sample())

        obs = self.env.render("rgb_array")
        for _ in range(self.frames_per_trans - 1 ):
            x, latent_dict = self._collect_datapoint(obs)
            self.append_to_trans_state(trans,x, latent_dict)
    
            # to_tensor true just in case policy is exactly a neural network
            action = self.policy(self.convert_fxn(x,to_tensor=True))
            obs, reward, done, _ = self.env.step(action)
            obs = self.env.render("rgb_array")
            self.append_to_trans_ard(trans,action,reward,done)
            
        
        x, latent_dict = self._collect_datapoint(obs)
        self.append_to_trans_state(trans,x, latent_dict)
        return trans
    
if __name__ == "__main__":
    env, action_space, grid_size,\
    num_directions, tot_examples, random_policy = setup_env("MiniGrid-Empty-8x8-v0")

    # "originalGame-v0"
    #"MiniGrid-Empty-8x8-v0"



    dc = DataCollector(policy=random_policy,env=env, frames_per_trans=5)
    trans = dc.collect_transition_per_the_policy()
    #     test_coords = (trans.x_coords[0], trans.y_coords[1])
    #     test_dir = trans.directions[0]
    #     spec_trans = dc.collect_specific_datapoint(test_coords,test_dir)

    #     assert (spec_trans.x_coords[0], spec_trans.y_coords[0])== test_coords
    #     assert spec_trans.directions[0] == test_dir

    from matplotlib import pyplot as plt

    #%matplotlib inline

    plt.imshow(trans.xs[0])
    
    
# import os
# import sys
# import numpy as np
# from matplotlib import pyplot as plt

# %matplotlib inline

# import data.custom_grids

# import gym

# #zip([97, 100, 32, 119, 115, None],["left","right","jump","up ladder", "down"])
# env = gym.make("originalGame-v0")
# nb_frames = 2
# done=False
# obs = env.reset()
# # init_x,init_y = env.env.game_state.game.newGame.Players[0].getPosition()
# # upd = np.random.choice([0,70,150])#np.random.randint(init_y)
# # env.env.game_state.game.newGame.Players[0].updateY(-1*upd)
# obs,reward, done, _ = env.step(3)
# for f in range(nb_frames):
#     if done: #check if the game is over
#         print("game over!")
#         env.reset()
#         init_x,init_y = env.env.game_state.game.newGame.Players[0].getPosition()
#         upd = np.random.choice([0,70,150])
#         env.env.game_state.game.newGame.Players[0].updateY(-1*upd)
        
    
#     player = env.env.game_state.game.newGame.Players[0]
#     pos, isj, ol = player.getPosition(), player.isJumping, player.onLadder
#     if f != -1:
#         plt.figure(f)
#         plt.imshow(obs,origin="upper")
#         plt.title("position: %.0f, %.0f; is_jumping: %i; on_ladder: %i"%(pos[0],pos[1],isj,ol))
#     action = np.random.choice(env.action_space.n)#actions[f]#
#     obs, reward, done, _ = env.step(action)
    