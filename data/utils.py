from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor, Grayscale
import numpy as np

import torch
from functools import partial
import math
import numpy as np
from collections import namedtuple
import copy

def convert_frame(obs, resize_to=(-1,-1),to_tensor=False,device="cpu"):
    pil_image = Image.fromarray(obs, 'RGB')
    
    transforms = [Resize(resize_to)] if resize_to != (-1,-1) else []
    if to_tensor:
        transforms.extend([ToTensor(),Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    transforms = Compose(transforms)
    frame = transforms(pil_image)
    if to_tensor:
        frame= frame.to(device)
    else:
        frame = np.asarray(frame)


    
    return frame

def convert_frames(frames,resize_to=(-1,-1),to_tensor=False,device="cpu"):
    convert = partial(convert_frame,resize_to=resize_to,to_tensor=to_tensor,device=device)
    return torch.stack([convert(frame) for frame in frames])


  
def make_empty_transition( args):
        Transition = get_transition_constructor(args)
        trans_list = [[] if k is not "state_param_dict" else {} for k in Transition._fields]
        trans = Transition(*trans_list)
        return trans

def get_transition_constructor(args):
        tuple_fields = ['xs']
        

        if args.there_are_actions:
            tuple_fields.append("actions")
        
        # add this last
        if args.needs_labels:
            tuple_fields.append("state_param_dict")
        
        Transition = namedtuple("Transition",tuple(tuple_fields))
        return Transition
               

def append_to_trans(trans,**kwargs):
        for k,v in kwargs.items():
            if k in trans._fields:
                trans._asdict()[k].append(copy.deepcopy(v))
    

def append_to_trans_param_dict(trans, param_dict):
        if "state_param_dict" in trans._fields:
            for k,v in param_dict.items():
                if k not in trans.state_param_dict:
                    trans.state_param_dict[k] = [copy.deepcopy(v)]
                else:
                    trans.state_param_dict[k].append(copy.deepcopy(v))