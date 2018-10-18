from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor, Grayscale
import numpy as np
from ple import gym_ple
import gym
import torch
from functools import partial
import math

def setup_env(args):
    env = gym.make(args.env_name)
    env.seed(args.seed)
    num_coord_buckets = args.buckets
    action_space = list(range(env.action_space.n))
    if hasattr(env.env, "ale"):
        def get_latent_dict(env):
            env_name = env.env.spec.id
            ram = env.env.ale.getRAM()
            if env_name == 'PrivateEye-v0':
                x_coord, y_coord = ram[63], ram[86]
            elif env_name == 'Pitfall-v0':
                x_coord, y_coord = ram[97], ram[105]
            else:
                assert False
        #     elif env_name == "MontezumaRevenge-v0":
        #         x_coord, y_coord = ram[42], ram[43]
            latent_dict = dict(x_coord=x_coord,y_coord=y_coord)
            return latent_dict
        env_name = args.env_name
        if env_name == 'PrivateEye-v0':
            num_x = 40
            num_y = 20
        elif env_name == 'Pitfall-v0':
            num_x = 20
            num_y = 20 
        else:
            assert False
        nclasses_table = dict(x_coord=num_x, y_coord=num_y)
    else:

        def get_latent_dict(env):
            grid_size = env.observation_space.shape[0]
            player = env.env.game_state.game.newGame.Players[0]
            (x_coord,y_coord), is_jumping, on_ladder = player.getPosition(), player.isJumping, player.onLadder
            x_thresh, y_thresh =  math.floor(grid_size/env.num_coord_buckets), math.floor(grid_size/env.num_coord_buckets)
            x_coord, y_coord =  math.floor(x_coord/x_thresh), math.floor(y_coord/y_thresh)

            latent_dict = dict(x_coord=x_coord,y_coord=y_coord, is_jumping=is_jumping, on_ladder=on_ladder)
            return latent_dict


        nclasses_table = dict(x_coord=num_coord_buckets, y_coord=num_coord_buckets,is_jumping=2, on_ladder=2 )
        
    env.num_coord_buckets = num_coord_buckets
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


