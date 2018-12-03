import numpy as np
from collections import namedtuple
import copy
from data.utils import setup_env, convert_frame
from data.utils import convert_frame
from functools import partial
from data.utils import setup_env

tuple_fields = ['xs','actions', 'rewards', 'dones', "state_param_dict"]


Transition = namedtuple("Transition",tuple(tuple_fields))


def make_empty_transition():
    transition_constructor = Transition
    num_fields = len(transition_constructor.__dict__["_fields"])
    trans_list = [[] for _ in range(num_fields -1)]
    trans_list.append({})
    trans = transition_constructor(*trans_list)
    return trans
    

class DataCollector(object):
    def __init__(self, args, policy=None):
        self.args = args
        self.convert_fxn = partial(convert_frame, resize_to=args.resize_to)
        self.env = setup_env(args)
        self.env.reset()
        #to avoid black frame
        self.env.step(self.env.action_space.n - 1)
        
        if policy:
            self.policy = policy
        else:
            rng = np.random.RandomState(args.seed)
            random_policy = lambda x0: rng.randint(self.env.action_space.n)
            self.policy=random_policy
        assert args.frames_per_trans >=2, "must have at least an s,a,s triplet"
        self.frames_per_trans = args.frames_per_trans

        
    def _collect_datapoint(self,obs):
        x = self.convert_fxn(obs)
        if "eval" in self.args.mode or "test" in self.args.mode:
            latent_dict = self.env.get_latent_dict(self.env)
        else:
            latent_dict = {}
        return x, latent_dict

    def append_to_trans_ard(self,trans,action,reward, done):
        trans.actions.append(copy.deepcopy(action))
        trans.rewards.append(copy.deepcopy(reward))
        trans.dones.append(copy.deepcopy(done))
        
    def append_to_trans_state(self,trans,x, latent_dict):
        trans.xs.append(copy.deepcopy(x))
        for k,v in latent_dict.items():
            if k not in trans.state_param_dict:
                trans.state_param_dict[k] = [copy.deepcopy(v)]
            else:
                trans.state_param_dict[k].append(copy.deepcopy(v))

        
        
    def collect_transition_per_the_policy(self):
        trans = make_empty_transition()
        
        #obs, reward, done, _ = self.env.step(env.action_space.sample())

        obs = self.env.render("rgb_array")
        for _ in range(self.frames_per_trans - 1 ):
            x, latent_dict = self._collect_datapoint(obs)
            self.append_to_trans_state(trans,x, latent_dict)
    
            # todo: be able to handle policy being nn that runs on cpu, so we can still collect data with multiprocessing solely on cpu while DEVICE/args.device is still equal to cuda
            action = self.policy(self.convert_fxn(x,to_tensor=False))
            obs, reward, done, _ = self.env.step(action)
            obs = self.env.render("rgb_array")
            self.append_to_trans_ard(trans,action,reward,done)
            
        
        x, latent_dict = self._collect_datapoint(obs)
        self.append_to_trans_state(trans,x, latent_dict)
        return trans
    
if __name__ == "__main__":
    env, action_space, grid_size,\
    num_directions, tot_examples, random_policy = setup_env("MiniGrid-Empty-8x8-v0")



    dc = DataCollector(policy=random_policy,env=env, frames_per_trans=5)
    trans = dc.collect_transition_per_the_policy()

    from matplotlib import pyplot as plt

    #%matplotlib inline

    plt.imshow(trans.xs[0])
    
    