import numpy as np
import torch
from functools import partial
import math
from data.env_utils import get_state_params



def setup_env(args): 
    if args.ple:
        from ple import gym_ple
    import gym
    env = gym.make(args.env_name)
    env.seed(args.seed) 
    env.num_buckets = args.buckets
    args.num_actions = env.action_space.n
    if "eval" in args.mode or "test" in args.mode:
        print(args.mode)
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