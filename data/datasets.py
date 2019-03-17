import numpy as np
import copy
from data.utils import convert_frame, convert_frames
from functools import partial
from torchvision.transforms import ToTensor,Compose,Resize,Normalize
from torchvision.datasets import ImageFolder
import json
from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset
import torch
from collections import defaultdict
from data.utils import appendabledict

class EpisodeDataset(Dataset):
    
    def __init__(self,
                 env,
                 frames_per_example=5,
                 max_frames=-1, 
                 resize_to=(-1,-1),
                 stride =1,
                 policy=None):
        
        self.convert_fxn = partial(convert_frame, resize_to=resize_to)
        #self.args = args
        self.stride = stride
        self.env = env #setup_env(args)
        self.resize_to = resize_to
        self.policy = policy
        self.frames_per_example = frames_per_example
        self.max_frames = max_frames + (frames_per_example * stride)  if max_frames != -1 else np.inf
        self.frames, self.actions, self.label_dict = self.collect_episode()
        self.inds = range(len(self.frames) - (self.frames_per_example * stride)) 
        
    def _step(self):
        if self.policy:
            action = self.policy(self.convert_fxn(x,to_tensor=False))
        else:
            action = self.env.action_space.sample()
            
        return action
        
           
    def collect_episode(self):
        label_dict = appendabledict(list)
        
        done = False
        obs, info = self.env.reset()
        frame = self.convert_fxn(self.env.render("rgb_array"))

        label_dict.append_update(info)
        frames = [frame]
        actions = []
        
        frame_count = 1
        while not done and frame_count < self.max_frames:
            action = self._step()
            obs, reward, done, info = self.env.step(action) 
            frame = self.env.render("rgb_array")
            label_dict.append_update(info)
            frame = self.convert_fxn(frame)
            frames.append(frame)
            actions.append(action)
            frame_count += 1
            
            
        for k,v in label_dict.items():
            label_dict[k] = np.stack(v)

        return np.stack(frames), np.stack(actions), label_dict
    
    def __getitem__(self, index):
        ind = self.inds[index]
        end_ind = ind + (self.frames_per_example * self.stride)
        action_end_ind = end_ind - 1
        slice_ = slice(ind,end_ind, self.stride)
        #action_slice_ = slice(ind,action_end_ind, self.stride)
        frames = self.frames[slice_]
        frames =  convert_frames(frames, resize_to=self.resize_to, to_tensor=True)
        frames = frames.transpose(0,1)
        #actions = self.actions[action_slice_]
        labels = self.label_dict.subslice(slice_)
        return frames,labels # actions #, labels

    def __len__(self):
        return len(self.inds)

    def __add__(self, other):
        return ConcatDataset([self, other])
    
        
class EnvDataset(ConcatDataset):
    def __init__(self, total_frames, **kwargs):
        max_frames = kwargs["max_frames"]
        num_episodes = total_frames // max_frames
        episodes = [ EpisodeDataset(**kwargs  ) for _ in range(num_episodes )]
        
        num_frames = sum([len(ep) for ep in episodes])
        while num_frames  < total_frames:
            kwargs["max_frames"] = total_frames - num_frames
            episodes.append(EpisodeDataset(**kwargs))
            num_frames = sum([len(ep) for ep in episodes])
        super(EnvDataset,self).__init__(episodes)