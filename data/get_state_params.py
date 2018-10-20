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
    assert coord >= min_coord
    coord_range = (max_coord - min_coord) + 1
    thresh =  coord_range/num_buckets
    bucketed_coord =  np.floor((coord - min_coord) /thresh)
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
    x_coord, y_coord = bucket_coord(x_coord,env.num_buckets, grid_size),bucket_coord(y_coord,env.num_buckets, grid_size)

    latent_dict = dict(x_coord=x_coord,y_coord=y_coord, is_jumping=is_jumping, on_ladder=on_ladder)
    return latent_dict

def monster_kong_get_nclasses_table(env):
    num_coord_buckets = env.num_buckets
    nclasses_table = dict(x_coord=num_coord_buckets, y_coord=num_coord_buckets,is_jumping=2, on_ladder=2 )
    return nclasses_table

def waterworld_get_latent_dict(env):
    len_x, len_y, _ = env.observation_space.shape
    creeps = env.env.game_state.game.creeps
    red_enemy = [c for c in creeps if c.TYPE == "BAD"][0]
    player = env.env.game_state.game.player
    x_coord, enemy_x = [bucket_coord(coord,env.num_buckets,len_x)\
                        for coord in [player.pos.x,red_enemy.pos.x ]]

    y_coord, enemy_y = [bucket_coord(coord,env.num_buckets,len_y)\
                        for coord in [player.pos.y,red_enemy.pos.y ]]

    latent_dict = dict(x_coord = x_coord, y_coord=y_coord,
                       enemy_x = enemy_x, enemy_y = enemy_y)
    return latent_dict

def waterworld_get_nclasses_table(env):
    return dict(x_coord = env.num_buckets, y_coord=env.num_buckets,
                   enemy_x = env.num_buckets, enemy_y = env.num_buckets)
    

def snake_get_latent_dict(env):
    food_x, food_y = env.env.game_state.game.food.pos.x, env.env.game_state.game.food.pos.y
    food_x, food_y = bucket_coord(food_x,20,64,-2), bucket_coord(food_y,20,64,-2)
    head_x, head_y = env.env.game_state.game.player.head.pos.x,env.env.game_state.game.player.head.pos.y
    head_x, head_y = bucket_coord(head_x,20,64,-2), bucket_coord(head_y,20,64,-2)
    latent_dict = dict(x_coord=head_x,y_coord=head_y,food_x=food_x,food_y=food_y)
    return latent_dict

def snake_get_nclasses_table(env):
    num_buckets = 20
    nclasses = dict(x_coord=num_buckets,y_coord=num_buckets,food_x=num_buckets,food_y=num_buckets)
    return nclasses


def flappybird_get_latent_dict(env):
    y_coord = env.env.game_state.game.player.pos_y
    y_coord = bucket_coord(y_coord,20,env.env.game_state.game.height)
    latent_dict = dict(y_coord=y_coord)
    return latent_dict

def flappybird_get_nclasses_table(env):
    return dict(y_coord=20)

def catcher_get_latent_dict(env):
    h,w = env.env.game_state.game.height, env.env.game_state.game.width
    x_coord = env.env.game_state.game.player.rect.centerx
    fruit_x, fruit_y = env.env.game_state.game.fruit.rect.centerx,\
                        env.env.game_state.game.fruit.rect.centery
    if fruit_y < 0 or fruit_y >= h:
        fruit_x, fruit_y = 0,0
    fruit_x = bucket_coord(fruit_x,20,w)
    fruit_y = bucket_coord(fruit_y,20,h)
    x_coord = bucket_coord(x_coord,20,w)
    latent_dict = dict(x_coord=x_coord,fruit_x=fruit_x,fruit_y=fruit_y)
    return latent_dict

def catcher_get_nclasses_table(env):
    return dict(x_coord=20,fruit_x=20,fruit_y=20)