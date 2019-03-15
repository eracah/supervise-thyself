from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor, Grayscale
import numpy as np

import torch
from functools import partial
import math
import numpy as np
from collections import namedtuple
import copy
from collections import defaultdict
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


class appendabledict(defaultdict):
    def __init__(self,type_,*args,**kwargs):
        self.type_ =  type_
        super().__init__(type_,*args,**kwargs)
        
    def subslice(self,slice_):
        """indexes every value in the dict according to a specified slice

        Parameters
        ----------
        slice : int or slice type
            An indexing slice , e.g., ``slice(2, 20, 2)`` or ``2``.


        Returns
        -------
        sliced_dict : dict (not appendabledict type!)
            A dictionary with each value from this object's dictionary, but the value is sliced according to slice_
            e.g. if this dictionary has {a:[1,2,3,4], b:[5,6,7,8]}, then self.subslice(2) returns {a:3,b:7}
                 self.subslice(slice(1,3)) returns {a:[2,3], b:[6,7]}

         """
        sliced_dict = {}
        for k,v in self.items():
            sliced_dict[k] = v[slice_]
        return sliced_dict
    
    def append_update(self,other_dict):
        """appends current dict's values with values from other_dict

        Parameters
        ----------
        other_dict : dict
            A dictionary that you want to append to this dictionary


        Returns
        -------
        Nothing. The side effect is this dict's values change

         """
        for k,v in other_dict.items():
            self.__getitem__(k).append(v)