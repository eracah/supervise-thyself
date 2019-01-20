from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor, Grayscale
import numpy as np
# from ple import gym_ple
# import gym
import torch
from functools import partial
import math

def bucket_coord(coord, num_buckets, max_coord, min_coord=0):
    try:
        assert (coord < max_coord and coord >= min_coord)
    except:
        print("coord: %i, max: %i, min: %i, num_buckets: %i"%(coord, max_coord, min_coord,num_buckets))
        assert False, coord
        #coord = 0
    coord_range = (max_coord - min_coord) + 1
    thresh =  coord_range/num_buckets
    bucketed_coord =  np.floor((coord - min_coord) /thresh)
    return bucketed_coord


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

def lunarlander_get_latent_dict(env):
    x,y = env.env.lander.position
    x_coord = bucket_coord(coord=x, num_buckets=env.num_buckets, max_coord=20.5)
    y_coord = bucket_coord(coord=y, num_buckets=env.num_buckets, max_coord=13.7)
    latent_dict = dict(x_coord=x_coord,y_coord=y_coord)
    return latent_dict

def lunarlander_get_nclasses_table(env):
    return dict(x_coord=env.num_buckets,
                y_coord=env.num_buckets)

def atari_get_latent_dict(env):
    env_name = env.env.spec.id
    ram = env.env.ale.getRAM()
    len_y, len_x, _ = env.observation_space.shape
    if env_name == 'PrivateEye-v0':
        x_coord, y_coord = ram[63], ram[86] #y_coord already bucketed to 40
        x_coord = bucket_coord(coord=x_coord,
                               num_buckets=env.num_buckets,
                               max_coord=len_x)
        latent_dict = dict(x_coord=x_coord,y_coord=y_coord)
    elif env_name == 'Pitfall-v0':
        x_coord= ram[97] # don't do y_coord its all messed up, y_coord = ram[105]
        x_coord = bucket_coord(coord=x_coord,
                               num_buckets=env.num_buckets,
                               max_coord=len_x)
        latent_dict = dict(x_coord=x_coord)
    else:
        assert False

    return latent_dict

def atari_get_nclasses_table(env):
    env_name  = env.env.spec.id
    if env_name == 'PrivateEye-v0':
        num_x = env.num_buckets
        num_y = env.num_buckets
        nclasses_table = dict(x_coord=num_x, y_coord=num_y)
    elif env_name == 'Pitfall-v0':
        num_x = env.num_buckets
        nclasses_table = dict(x_coord=num_x)
    else:
        assert False

    return nclasses_table


def monster_kong_get_latent_dict(env):
    grid_size = env.observation_space.shape[0]
    player = env.env.game_state.game.newGame.Players[0]
    (x_coord,y_coord), is_jumping, on_ladder = player.getPosition(), player.isJumping, player.onLadder
    x_coord, y_coord = bucket_coord(x_coord,env.num_buckets, grid_size),bucket_coord(y_coord,env.num_buckets, grid_size)

    latent_dict = dict(x_coord=x_coord,y_coord=y_coord, is_jumping=is_jumping, on_ladder=on_ladder)
    return latent_dict

def monster_kong_get_nclasses_table(env):
    nclasses_table = dict(x_coord=env.num_buckets, y_coord=env.num_buckets,is_jumping=2, on_ladder=2 )
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
    bc = partial(bucket_coord, num_buckets=env.num_buckets, max_coord=64, min_coord=-2)
    
    food_x, food_y = env.env.game_state.game.food.pos.x,\
                     env.env.game_state.game.food.pos.y
    food_x, food_y = bc(coord=food_x), bc(coord=food_y)
    
    head_x, head_y = env.env.game_state.game.player.head.pos.x,\
                     env.env.game_state.game.player.head.pos.y
    head_x, head_y = bc(coord=head_x), bc(coord=head_y)
    
    latent_dict = dict(x_coord=head_x,y_coord=head_y,food_x=food_x,food_y=food_y)
    return latent_dict

def snake_get_nclasses_table(env):
    nclasses = dict(x_coord=env.num_buckets,
                    y_coord=env.num_buckets,
                    food_x=env.num_buckets,
                    food_y=env.num_buckets)
    return nclasses


def flappybird_get_latent_dict(env):
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
    return latent_dict

def flappybird_get_nclasses_table(env):
    return dict(y_coord=env.num_buckets,pipe_x_coord=env.num_buckets)

flappybirdday_get_latent_dict = flappybirdnight_get_latent_dict = flappybird_get_latent_dict
flappybirdday_get_nclasses_table = flappybirdnight_get_nclasses_table = flappybird_get_nclasses_table

def catcher_get_latent_dict(env):
    h,w = env.env.game_state.game.height, env.env.game_state.game.width
    x_coord = env.env.game_state.game.player.rect.centerx
    fruit_x, fruit_y = env.env.game_state.game.fruit.rect.centerx,\
                        env.env.game_state.game.fruit.rect.centery
    if fruit_y < 0 or fruit_y >= h:
        fruit_x, fruit_y = 0,0
    fruit_x = bucket_coord(fruit_x,env.num_buckets,w)
    fruit_y = bucket_coord(fruit_y,env.num_buckets,h)
    x_coord = bucket_coord(x_coord,env.num_buckets,w)
    latent_dict = dict(x_coord=x_coord,fruit_x=fruit_x,fruit_y=fruit_y)
    return latent_dict

def catcher_get_nclasses_table(env):
    return dict(x_coord=20,fruit_x=env.num_buckets,fruit_y=env.num_buckets)