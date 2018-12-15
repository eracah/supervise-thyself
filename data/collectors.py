import numpy as np
from collections import namedtuple
import copy
from data.utils import setup_env, convert_frame
from data.utils import convert_frame
from functools import partial
from data.utils import setup_env

class EpisodeCollector(object):
    def __init__(self, args, policy=None):            
        self.convert_fxn = partial(convert_frame, resize_to=args.resize_to)
        self.args = args
        self.env = setup_env(args)
        self.env.reset()        
        if policy:
            self.policy = policy
        else:
            rng = np.random.RandomState(args.seed)
            random_policy = lambda x0: rng.randint(self.env.action_space.n)
            self.policy=random_policy

        
    def make_empty_transition(self):
        Transition = self.get_transition_constructor(self.args)
        num_fields = len(Transition._fields)
        trans_list = [[] for _ in range(num_fields)]

        trans = Transition(*trans_list)
        if "state_param_dict" in Transition._fields:
            trans._asdict()["state_param_dict"] = {}
        return trans
    
    @classmethod
    def get_transition_constructor(self, args):
        tuple_fields = ['xs']
        

        if args.there_are_actions:
            tuple_fields.append("actions")
        
        # add this last
        if args.mode == "eval" or args.mode == "test":
            tuple_fields.append("state_param_dict")
        
        Transition = namedtuple("Transition",tuple(tuple_fields))
        return Transition
               

    def append_to_trans(self,trans,**kwargs):
        for k,v in kwargs.items():
            if k in trans._fields:
                trans._asdict()[k].append(copy.deepcopy(v))
        
    def append_to_trans_param_dict(self,trans):
        if "state param_dict" in trans._fields:
            param_dict = self.env.get_latent_dict(self.env)
            for k,v in param_dict.items():
                if k not in trans.state_param_dict:
                    trans.state_param_dict[k] = [copy.deepcopy(v)]
                else:
                    trans.state_param_dict[k].append(copy.deepcopy(v))

        
        
    def collect_episode_per_the_policy(self,max_frames=-1):
        trans = self.make_empty_transition()
        done = False
        self.env.reset()
        obs = self.env.render("rgb_array")
        frame_count = 0
        while not done:
            x = self.convert_fxn(obs)
            self.append_to_trans_param_dict(trans)
            self.append_to_trans(trans,xs=x)
            action = self.policy(self.convert_fxn(x,to_tensor=False))
            obs, reward, done, _ = self.env.step(action)
            obs = self.env.render("rgb_array")
            self.append_to_trans(trans, actions=action, rewards=reward)
            frame_count += 1
            if max_frames is not -1 and frame_count >= max_frames:
                break
                
        x = self.convert_fxn(obs)
        self.append_to_trans_param_dict(trans)
        self.append_to_trans(trans,xs=x)
        return trans
    
if __name__ == "__main__":
    env, action_space, grid_size,\
    num_directions, tot_examples, random_policy = setup_env("MiniGrid-Empty-8x8-v0")



    dc = DataCollector(policy=random_policy,env=env, frames_per_trans=5)
    trans = dc.collect_transition_per_the_policy()

    from matplotlib import pyplot as plt

    #%matplotlib inline

    plt.imshow(trans.xs[0])
    
    