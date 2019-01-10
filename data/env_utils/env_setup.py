import numpy as np
import torch
from functools import partial
import math
from data.env_utils import get_state_params
import gym

# I got this from https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()
    
    def sonicify_action(self,a):
        return self.action(a)
        
    

def setup_env(args): 
    if args.retro:
        import retro
        env = SonicDiscretizer(retro.make(game=args.env_name, state=args.level))
        # 10 actions to pick from you can do from 0 to 4 at once and subtract invalid ones like UP and DOWN together or LEFT and RIGHT together
    
    else:
        if args.ple:
            from ple import gym_ple
        import gym
        env = gym.make(args.env_name)
    args.num_actions = env.action_space.n
    
    env.seed(args.seed) 
    env.num_buckets = args.buckets

    if args.needs_labels:
        add_labels_to_env(env,args)     
        args.nclasses_table = env.nclasses_table
    
    try:
        print(env.spec.id, args.env_name)
    except:
        try:
            print(env.gamename,env.statename, args.env_name, args.level)
        except:
            print(env.env.gamename,env.env.statename, args.env_name, args.level)
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