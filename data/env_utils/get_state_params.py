from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor, Grayscale
import numpy as np
# from ple import gym_ple
# import gym
import torch
from functools import partial
import math




def sonic_get_latent_dict(env):
    
    y = env.env.data.lookup_value("y")
    screen_y = env.env.data.lookup_value("screen_y")
    x = env.env.data.lookup_value("x")
    screen_x = env.env.data.lookup_value("screen_x")
    x_coord = bucket_coord(x - screen_x,env.num_buckets,200)
    y_coord = bucket_coord(y - screen_y,env.num_buckets,220)
    latent_dict = dict(x_coord=x_coord,y_coord=y_coord)
    return latent_dict

def sonic_get_nclasses_table(env):
    return dict(x_coord=env.num_buckets,
                y_coord=env.num_buckets)


