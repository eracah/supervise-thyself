import numpy as np
from collections import namedtuple
import copy
from data.utils import convert_frame
from functools import partial
from data.env_utils.env_setup import setup_env
from data.utils import make_empty_transition, append_to_trans, append_to_trans_param_dict

class EpisodeCollector(object):
    def __init__(self, args, policy=None):            
        self.convert_fxn = partial(convert_frame, resize_to=args.resize_to, device=args.device)
        self.args = args
        self.env = setup_env(args)
        self.env.reset()        

        self.policy = policy
#         else:
#             rng = np.random.RandomState(args.seed)
#             random_policy = lambda x0: rng.randint(self.env.action_space.n)
#             self.policy=random_policy
       
    def collect_episode_per_the_policy(self,max_frames=-1):
        trans = make_empty_transition(self.args)
        done = False
        self.env.reset()
        obs = self.env.render("rgb_array")
        frame_count = 0
        while not done:
            x = self.convert_fxn(obs)
            if "state_param_dict" in trans._fields:
                param_dict = self.env.get_latent_dict(self.env)
                append_to_trans_param_dict(trans, param_dict)
            append_to_trans(trans,xs=x)
            if self.policy:
                action = self.policy(self.convert_fxn(x,to_tensor=False))
            else:
                action = self.env.action_space.sample()


#             if hasattr(self.env,"sonicify_action"):
#                 sonic_action = self.env.sonicify_action(action)
#                 sonic_action = sonic_action.astype("int")
#                 print(sonic_action)
#                 obs, reward, done, _ = self.env.step(sonic_action)
#             else:
            obs, reward, done, _ = self.env.step(action)
            obs = self.env.render("rgb_array")
            append_to_trans(trans, actions=action, rewards=reward)
            frame_count += 1
            if max_frames is not -1 and frame_count >= max_frames:
                break
                
        x = self.convert_fxn(obs)
        if "state_param_dict" in trans._fields:
            param_dict = self.env.get_latent_dict(self.env)
            append_to_trans_param_dict(trans, param_dict)
        append_to_trans(trans,xs=x)
        return trans
    
    
    