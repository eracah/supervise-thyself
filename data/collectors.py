from utils import convert_frame
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid
import numpy as np
from collections import namedtuple
import copy

def get_trans_tuple():
        tuple_fields = ['xs','actions', 'rewards', 'dones']
        

        tuple_fields.extend(['x_coords', 'y_coords',"directions"])

        Transition = namedtuple("Transition",tuple(tuple_fields))
        return Transition

def make_empty_transition():
    transition_constructor = get_trans_tuple()
    num_fields = len(transition_constructor.__dict__["_fields"])
    trans_list = [[] for _ in range(num_fields)]
    trans = transition_constructor(*trans_list)
    return trans
    

class DataCollector(object):
    def __init__(self, policy=lambda x0: np.random.choice(3),
                        env=gym.make("MiniGrid-Empty-6x6-v0"),
                        convert_fxn=convert_frame,
                        frames_per_trans=2):
        self.convert_fxn = convert_fxn
        self.policy = policy
        self.env = env
        # datapoints_per_trans=0 means you just collect the current state and don't take an action
        # and get a new state
        assert frames_per_trans >=2, "must have at least an s,a,s triplet"
        self.frames_per_trans = frames_per_trans

        
    def _collect_datapoint(self):
        x = self.convert_fxn(self.env.render("rgb_array"))
        x_coord, y_coord = int(self.env.agent_pos[0]), int(self.env.agent_pos[1])
        direction = self.env.agent_dir
        return x, x_coord, y_coord, direction

    def append_to_trans_ard(self,trans,action,reward, done):
        trans.actions.append(copy.deepcopy(action))
        trans.rewards.append(copy.deepcopy(reward))
        trans.dones.append(copy.deepcopy(done))
        
    def append_to_trans_state(self,trans,x, x_coord, y_coord, direction):
        trans.xs.append(copy.deepcopy(x))
        trans.x_coords.append(copy.deepcopy(x_coord))
        trans.y_coords.append(copy.deepcopy(y_coord))
        trans.directions.append(copy.deepcopy(direction))
        
        
    def collect_transition_per_the_policy(self):
        trans = make_empty_transition()
        for _ in range(self.frames_per_trans - 1 ):
            x, x_coord, y_coord, direction = self._collect_datapoint()
            self.append_to_trans_state(trans,x, x_coord, y_coord, direction)
    
            # to_tensor true just in case policy is exactly a neural network
            action = self.policy(self.convert_fxn(x,to_tensor=True))
            _, reward, done, _ = self.env.step(action)
            
            self.append_to_trans_ard(trans,action,reward,done)
            
        
        x, x_coord, y_coord, direction = self._collect_datapoint()
        self.append_to_trans_state(trans,x, x_coord, y_coord, direction)
        return trans
    
    def collect_specific_datapoint(self,coords, direction):
        trans = make_empty_transition()
        self._get_desired_position(coords)
        self._get_desired_direction(direction)
        x, x_coord, y_coord, direction  = self._collect_datapoint()
        self.append_to_trans_state(trans,x, x_coord, y_coord, direction)
        return trans
    
    def _get_desired_position(self, coords):
        self.env.agent_pos = np.asarray(coords)
        
    def _get_desired_direction(self,desired_direction):
        true_direction = self.env.agent_dir
        while not np.allclose(true_direction,desired_direction):
            _ = self.env.step(0)
            true_direction = self.env.agent_dir

if __name__ == "__main__":
    dc = DataCollector(datapoints_per_trans=5)

    trans = dc.collect_transition_per_the_policy()
    test_coords = (trans.x_coords[0], trans.y_coords[1])
    test_dir = trans.directions[0]
    spec_trans = dc.collect_specific_datapoint(test_coords,test_dir)
    
    assert (spec_trans.x_coords[0], spec_trans.y_coords[0])== test_coords
    assert spec_trans.directions[0] == test_dir

