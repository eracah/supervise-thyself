import numpy as np
import torch
from functools import partial
import math
import gym
import retro
from data.env_utils import wrappers

        
def setup_env(env_name, seed=0):
    gym_mod = get_gym_module(env_name)
    env = gym_mod.make(env_name)
    env.seed(seed)
    wrapper = get_wrapper(env, env_name)
    env = wrapper(env)
    return env
    
        
def get_wrapper(env, env_name):
    env_type = get_env_type(env,env_name)
    wrapper = getattr(wrappers, env_type + "Wrapper")
    return wrapper
    
def get_gym_module(env_name):
    if "FlappyBird" in env_name:
        from ple import gym_ple
    module = gym if "Sonic" not in env_name else retro
    return module
    
def get_env_type(env, env_name):
    if "Sonic" in env_name:
        env_type = "Sonic"
    
    elif hasattr(env.unwrapped, "ale"):
        env_type = "Atari"
    
    else:
        env_type = env_name.split("-")[0]
    return env_type