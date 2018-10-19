from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor, Grayscale
import numpy as np
from ple import gym_ple
import gym
import torch
from functools import partial
import math

def bucket_coord(coord,num_buckets, max_coord, min_coord=0):
    assert coord < max_coord
    assert coord > min_coord
    coord_range = (max_coord - min_coord) + 1
    thresh =  np.floor(coord_range/num_buckets)
    bucketed_coord =  np.floor(coord/thresh)
    return bucketed_coord


def atari_get_latent_dict(env):
    env_name = env.env.spec.id
    ram = env.env.ale.getRAM()
    len_y, len_x, _ = env.observation_space.shape
    if env_name == 'PrivateEye-v0':
        x_coord, y_coord = ram[63], ram[86]
        #y_coord already bucketed to 40
        x_coord = bucket_coord(x_coord,40,len_x)
    elif env_name == 'Pitfall-v0':
        x_coord, y_coord = ram[97], ram[105]
        x_coord = bucket_coord(x_coord,40, max_coord=len_x, min_coord=0)
        y_coord = bucket_coord(y_coord,20, max_coord=len_y, min_coord=0)
    else:
        assert False
    latent_dict = dict(x_coord=x_coord,y_coord=y_coord)
    return latent_dict

def atari_get_nclasses_table(env):
    env_name  = env.env.spec.id
    if env_name == 'PrivateEye-v0':
        num_x = 40
        num_y = 40
    elif env_name == 'Pitfall-v0':
        num_x = 40
        num_y = 20 
    else:
        assert False
    nclasses_table = dict(x_coord=num_x, y_coord=num_y)
    return nclasses_table


def monster_kong_get_latent_dict(env):
    grid_size = env.observation_space.shape[0]
    player = env.env.game_state.game.newGame.Players[0]
    (x_coord,y_coord), is_jumping, on_ladder = player.getPosition(), player.isJumping, player.onLadder
    x_thresh, y_thresh =  math.floor(grid_size/env.num_coord_buckets), math.floor(grid_size/env.num_coord_buckets)
    x_coord, y_coord =  math.floor(x_coord/x_thresh), math.floor(y_coord/y_thresh)

    latent_dict = dict(x_coord=x_coord,y_coord=y_coord, is_jumping=is_jumping, on_ladder=on_ladder)
    return latent_dict

def monster_kong_get_nclasses_table(env):
    num_coord_buckets = env.num_coord_buckets
    nclasses_table = dict(x_coord=num_coord_buckets, y_coord=num_coord_buckets,is_jumping=2, on_ladder=2 )
    return nclasses_table

def waterworld_get_latent_dict(env):
    len_x, len_y, _ = env.observation_space.shape
    _,_, red_enemy  = list(env.env.game_state.game.creeps)
    assert red_enemy.TYPE == 'BAD'
    player = env.env.game_state.game.player
    x_coord, enemy_x = [bucket_coord(coord,args.num_buckets,len_x)\
                        for coord in [player.pos.x,red_enemy.pos.x ]]

    y_coord, enemy_y = [bucket_coord(coord,args.num_buckets,len_y)\
                        for coord in [player.pos.y,red_enemy.pos.y ]]

    latent_dict = dict(x_coord = x_coord, y_coord=y_coord,
                       enemy_x = enemy_x, enemy_y = enemy_y)
    return latent_dict


def catcher_get_latent_dict(env):
    pass

def setup_env(args):
    env = gym.make(args.env_name)
    env.seed(args.seed)
    num_coord_buckets = args.buckets
    env.num_coord_buckets = num_coord_buckets
    action_space = list(range(env.action_space.n))
    if hasattr(env.env, "ale"):
        get_latent_dict = atari_get_latent_dict
        nclasses_table = atari_get_nclasses_table(env)
    elif args.env_name in ['originalGame-v0','nosemantics-v0','noobject-v0','nosimilarity=v0','noaffordance-v0']:
        get_latent_dict = monster_kong_get_latent_dict
        nclasses_table = monster_kong_get_nclasses_table(env)
    else:
        raise NotImplementedError

    env.get_latent_dict = get_latent_dict
    env.nclasses_table = nclasses_table
    return env


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


