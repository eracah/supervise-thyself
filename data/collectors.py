import numpy as np
from collections import namedtuple
import copy
from data.utils import setup_env, convert_frame
from data.utils import convert_frame
from functools import partial
from data.utils import setup_env

class EpisodeCollector(object):
    def __init__(self, args, policy=None):
        self.tuple_fields = ['xs']
        self.args = args
        if args.mode == "test" or args.mode == "test":
            self.tuple_fields.append("state_param_dict")
        if args.model_name == "inv_model" or args.model_name == "pred_frames":
            self.tuple_fields.append("actions")
            
        self.convert_fxn = partial(convert_frame, resize_to=args.resize_to)
        self.env = setup_env(args)
        self.env.reset()        
        if policy:
            self.policy = policy
        else:
            rng = np.random.RandomState(args.seed)
            random_policy = lambda x0: rng.randint(self.env.action_space.n)
            self.policy=random_policy

        
    def make_empty_transition(self):
        transition_constructor = self.get_transition_constructor()
        num_fields = len(transition_constructor.__dict__["_fields"])
        trans_list = [[] for _ in range(num_fields)]
        if "state_param_dict" in self.tuple_fields:
            trans_list[-1] = {}
        trans = transition_constructor(*trans_list)
        return trans
    
    def get_transition_constructor(self):
        Transition = namedtuple("Transition",tuple(self.tuple_fields))
        return Transition
    
    def _collect_datapoint(self,obs):
        x = self.convert_fxn(obs)
        return x
    
    def _collect_param_dict(self):
        param_dict = self.env.get_latent_dict(self.env)
        return param_dict
        

    def append_to_trans(self,trans,**kwargs):
        for k,v in kwargs.items():
            trans._asdict()[k].append(copy.deepcopy(v))
        
    def append_to_trans_param_dict(self,trans, param_dict):
        for k,v in param_dict.items():
            if k not in trans.state_param_dict:
                trans.state_param_dict[k] = [copy.deepcopy(v)]
            else:
                trans.state_param_dict[k].append(copy.deepcopy(v))

        
        
    def collect_episode_per_the_policy(self):
        trans = self.make_empty_transition()
        done = False
        self.env.reset()
        obs = self.env.render("rgb_array")
        while not done:
            x = self._collect_datapoint(obs)
            if "state param_dict" in self.tuple_fields:
                param_dict = self._collect_param_dict()
                self.append_to_trans_param_dict(self,trans, param_dict)
            self.append_to_trans(trans,xs=x)
    
            action = self.policy(self.convert_fxn(x,to_tensor=False))
            obs, reward, done, _ = self.env.step(action)
            obs = self.env.render("rgb_array")
            if "actions" in self.tuple_fields:
                self.append_to_trans(trans,actions=action)
            if "rewards" in self.tuple_fields:
                 self.append_to_trans(trans,rewards=reward)
                
            
        x = self._collect_datapoint(obs)
        
        if "state param_dict" in self.tuple_fields:
            param_dict = self._collect_param_dict()
            self.append_to_trans_param_dict(self,trans, param_dict)
        
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
    
    