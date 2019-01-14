from collections import namedtuple
import torch
import numpy as np
import random
from functools import partial
import copy
from data.utils import convert_frames
import math
from data.utils import make_empty_transition, append_to_trans, append_to_trans_param_dict, get_transition_constructor

class DataSampler(object):
    """buffer of episodes. you can sample it like a true replay buffer (with replacement) using self.sample
    or like normal data iterator used in most supervised learning problems with sellf.__iter__()"""
    """Memory is uint8 to save space, then when you sample it converts to float tensor"""
    """Note no matter what the frames_per_example is, each frame ind (except the ones)"""
    def __init__(self,args, batch_size=64):
        self.args = args
        self.Transition = get_transition_constructor(self.args)
        self.stride = self.args.stride
        self.num_frames = self.args.frames_per_example
        self.DEVICE = self.args.device
        self.batch_size = batch_size
        self.episodes = []
        self.all_inds = None
   
    def push(self, episode_trans):
        """Saves a transition."""
        self.episodes.append(episode_trans)

    def sample(self,batch_size=None, with_replacement=False):
        batch_size = self.batch_size if batch_size is None else batch_size
        
        #sample indices into frames
        all_inds = self.get_all_inds()
        all_arr = np.stack(random.choices(all_inds, k=batch_size))
        ep_inds, frame_inds = all_arr[:,0], all_arr[:,1]
        
        raw_sample = self.raw_sample(ep_inds, frame_inds)
        batch = self._convert_raw_sample(raw_sample)
        return batch

    
    def get_all_inds(self):
        if self.all_inds is None:
            ep_lens = {i:len(self.episodes[i].xs) for i in  range(self.num_episodes)}

           # print([(ep_ind, frame_ind) for frame_ind in range(ep_lens[ep_ind] - self.stride)])
            all_possible_inds = np.concatenate([[(ep_ind, frame_ind) 
                                                # if the sample is k consecutive frames then we can only use frame indices that at most k frames from the end of the episode
                                               for frame_ind in range(ep_lens[ep_ind] - self.num_frames  + 1 )] 
                                               for ep_ind in range(self.num_episodes)])
            all_inds = all_possible_inds
        else:
            all_inds = self.all_inds
        return all_inds

    
    def raw_sample(self, ep_inds, frame_inds):
        transitions = [self._sample(ep_ind,frame_ind, self.num_frames, self.stride) for 
                                    ep_ind, frame_ind in zip(ep_inds,frame_inds)]
        return transitions
        
    
    def _sample(self,ep_ind,frame_ind, num=1, stride=1):
        """given an episode index, take that episode and starting at the frame_indth frame, sample frames from
        range(frame_ind, num, stride)""" 
        ep = self.episodes[ep_ind]
        frames_until_end = len(ep.xs)  - frame_ind 
        frames_needed_from_start_ind = num*stride
        if frames_until_end < frames_needed_from_start_ind:
            assert False, "end of episode reached and we didn't get all da frames yet!"
            
        trans = make_empty_transition(self.args)
        for _ in range(num - 1):
            frame_kwargs = {k:ep._asdict()[k][frame_ind] for k in ep._asdict().keys() if k is not "state_param_dict" }
            append_to_trans(trans, **frame_kwargs)
            if "state_param_dict" in ep._fields:
                param_dict = {k:ep.state_param_dict[k][frame_ind] for k in ep.state_param_dict.keys()}
                append_to_trans_param_dict(trans,param_dict)
            frame_ind += stride
                              
                              
        frame = ep.xs[frame_ind]
        append_to_trans(trans,xs=frame)
        if "state_param_dict" in ep._fields:
                param_dict = {k:ep.state_param_dict[k][frame_ind] for k in ep.state_param_dict.keys()}
                append_to_trans_param_dict(trans,param_dict)
        return trans
    
    def _convert_raw_sample(self,transitions):
        """converts 8-bit RGB to float and pytorch tensor"""
        # puts all trans objects into one trans object
        trans = self._combine_transitions_into_one_big_one(transitions)
        batch = self._convert_fields_to_pytorch_tensors(trans)
        return batch
        
    def _combine_transitions_into_one_big_one(self,transitions):
        fields = []
        for i,field in enumerate(zip(*transitions)):
            if isinstance(field[0],list):
                new_field = np.stack([list_ for list_ in field])
                if str(new_field.dtype) == "bool":
                    new_field = new_field.astype("int")
                #print(field.shape,field)
            if isinstance(field[0],dict):
                new_field = {}
                for k in field[0].keys():
                    all_items_of_key_k = [dic[k] for dic in field]
                    array_of_items_of_key_k = np.stack([list_ for list_ in all_items_of_key_k])
                    new_field[k] = array_of_items_of_key_k

            fields.append(new_field)
        
        return self.Transition(*fields)
    
    def _convert_fields_to_pytorch_tensors(self,trans):
        tb_dict = trans._asdict()
        if "state_param_dict" in tb_dict:
            for k,v  in trans.state_param_dict.items():
                tb_dict["state_param_dict"][k] = torch.tensor(v).to(self.DEVICE)
                
        convert_fxn = partial(convert_frames, to_tensor=True,device=self.DEVICE)
                              
        tb_dict["xs"] = torch.stack([convert_fxn(np.asarray(trans.xs[i])) for
                                                     i in range(len(trans.xs))]) #.to(self.DEVICE)

        if "actions" in tb_dict:
            tb_dict["actions"] = torch.from_numpy(np.asarray(trans.actions)).to(self.DEVICE)
        if "rewards" in tb_dict:
            tb_dict["rewards"] = torch.from_numpy(np.asarray(trans.rewards)).to(self.DEVICE)

        
        
        batch = self.Transition(*list(tb_dict.values()))
        return batch

    def __iter__(self):
        """Iterator that samples without replacement for replay buffer
        It's basically like a standard sgd setup
        If you want to sample with replacement like standard replay buffer use self.sample"""
        all_inds = self.get_all_inds()
        random.shuffle(all_inds)
        size = len(all_inds)
        for st in range(0, size, self.batch_size):
            end = st+self.batch_size if st+self.batch_size <= size else size
            batch_inds = np.stack(all_inds[st:end])
            ep_inds, frame_inds = batch_inds[:,0], batch_inds[:,1]
            raw_sample = self.raw_sample(ep_inds, frame_inds)
            yield self._convert_raw_sample(raw_sample)
    
    def __len__(self):
        return len(self.episodes)
    
    @property
    def num_episodes(self):
        return self.__len__()
    

    
    

    