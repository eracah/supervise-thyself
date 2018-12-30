import numpy as np
import torch
from functools import partial
import math
from data.env_utils import get_state_params



def setup_env(args): 
    if args.retro:
        import retro
        env = retro.make(game=args.env_name, state=args.level)
        args.num_actions = 126 # 10 actions to pick from you can do from 0 to 4 at once and subtract invalid ones like UP and DOWN together or LEFT and RIGHT together
    
    else:
        if args.ple:
            from ple import gym_ple
        import gym
        env = gym.make(args.env_name)
        args.num_actions = env.action_space.n
    
    env.seed(args.seed) 
    env.num_buckets = args.buckets

    if args.task == "infer" or (args.mode == "test" and args.task == "predict"):
        add_labels_to_env(env,args)     
        args.nclasses_table = env.nclasses_table

    return env

def add_labels_to_env(env, args):
    if hasattr(env.env, "ale"):
        get_latent_dict = get_state_params.atari_get_latent_dict
        nclasses_table = get_state_params.atari_get_nclasses_table(env)
    elif args.env_name in ['originalGame-v0','nosemantics-v0','noobject-v0','nosimilarity-v0','noaffordance-v0']:
        get_latent_dict = get_state_params.monster_kong_get_latent_dict
        nclasses_table = get_state_params.monster_kong_get_nclasses_table(env)
    else:
        try:
            get_latent_dict = getattr(get_state_params,env.spec.id.strip("-v0").lower() + "_get_latent_dict")
            nclasses_table = getattr(get_state_params,env.spec.id.strip("-v0").lower() + "_get_nclasses_table")(env)
        except:
            raise NotImplementedError

    env.get_latent_dict = get_latent_dict
    env.nclasses_table = nclasses_table