from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor, Grayscale
import numpy as np
import data.custom_grids
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid
import torch
from functools import partial
import math

def setup_env(env_name, num_coord_buckets=20, seed=4):
    env = gym.make(env_name)
    env.seed(seed)
    if "MiniGrid" in env_name:
        env.action_space = gym.spaces.discrete.Discrete(3)
        grid_size = env.grid_size - 2
        if grid_size < num_coord_buckets:
            num_coord_buckets = grid_size
        num_directions = 4
        def get_latent_dict(env):
            x_coord, y_coord  = int(env.agent_pos[0]), int(env.agent_pos[1])
            #print(x_coord,y_coord,env.num_coord_buckets)
            x_thresh, y_thresh =  math.floor((env.grid_size - 2)/env.num_coord_buckets), math.floor((env.grid_size - 2)/env.num_coord_buckets)
            x_coord, y_coord =  math.floor(x_coord/x_thresh), math.floor(y_coord/y_thresh)
                
            #print(x_coord,y_coord)
            direction = env.agent_dir
            latent_dict = dict(x_coord=x_coord, y_coord=y_coord, direction=direction)
            return latent_dict
        
        nclasses_table = dict(x_coord=num_coord_buckets, y_coord=num_coord_buckets,direction=num_directions )
        
        

    else:
        action_space = list(range(env.action_space.n))
        #grid_size = env.observation_space.shape[0]
        num_directions = None
        tot_examples = None
        def get_latent_dict(env):
            grid_size = env.observation_space.shape[0]
            player = env.env.game_state.game.newGame.Players[0]
            (x_coord,y_coord), is_jumping, on_ladder = player.getPosition(), player.isJumping, player.onLadder
            #print(x_coord,y_coord,env.num_coord_buckets)
            x_thresh, y_thresh =  math.floor(grid_size/env.num_coord_buckets), math.floor(grid_size/env.num_coord_buckets)
            x_coord, y_coord =  math.floor(x_coord/x_thresh), math.floor(y_coord/y_thresh)
            #print(x_coord,y_coord)
            latent_dict = dict(x_coord=x_coord,y_coord=y_coord, is_jumping=is_jumping, on_ladder=on_ladder)
            return latent_dict
        

        nclasses_table = dict(x_coord=num_coord_buckets, y_coord=num_coord_buckets,is_jumping=2, on_ladder=2 )
        
    num_actions = env.action_space.n
    env.num_coord_buckets = num_coord_buckets
    env.get_latent_dict = get_latent_dict
    env.nclasses_table = nclasses_table
    
    rng = np.random.RandomState(seed)
    random_policy = lambda x0: rng.randint(num_actions)
    return env, random_policy


def convert_frame(obs, resize_to=(-1,-1),to_tensor=False):
    pil_image = Image.fromarray(obs, 'RGB')
    
    transforms = [Resize(resize_to)] if resize_to != (-1,-1) else []
    if to_tensor:
        transforms.extend([ToTensor(),Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    transforms = Compose(transforms)
    frame = transforms(pil_image)
    if to_tensor:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        frame= frame.to(DEVICE)
    else:
        frame = np.asarray(frame)


    
    return frame

def convert_frames(frames,resize_to=(64,64),to_tensor=False):
    convert = partial(convert_frame,resize_to=resize_to,to_tensor=to_tensor)
    return torch.stack([convert(frame) for frame in frames])