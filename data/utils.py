from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor, Grayscale
import numpy as np
from ple import gym_ple
import gym
import torch
from functools import partial
import math
from data import get_state_params



def setup_env(args):   
    env = gym.make(args.env_name)
    env.seed(args.seed) 
    env.num_buckets = args.buckets
    action_space = list(range(env.action_space.n))
    args.num_actions = env.action_space.n
    if "eval" in args.mode or "test" in args.mode:
        print(args.mode)
        add_labels_to_env(env,args)

    return env

def add_labels_to_env(env, args):
    if hasattr(env.env, "ale"):
        get_latent_dict = get_state_params.atari_get_latent_dict
        nclasses_table = get_state_params.atari_get_nclasses_table(env)
    elif args.env_name in ['originalGame-v0','nosemantics-v0','noobject-v0','nosimilarity=v0','noaffordance-v0']:
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
def convert_frame(obs, resize_to=(-1,-1),to_tensor=False):
    pil_image = Image.fromarray(obs, 'RGB')
    
    transforms = [Resize(resize_to)] if resize_to != (-1,-1) else []
    if to_tensor:
        transforms.extend([ToTensor(),Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    transforms = Compose(transforms)
    frame = transforms(pil_image)
    if to_tensor:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        frame= frame.to(DEVICE)
    else:
        frame = np.asarray(frame)


    
    return frame

def convert_frames(frames,resize_to=(64,64),to_tensor=False):
    convert = partial(convert_frame,resize_to=resize_to,to_tensor=to_tensor)
    return torch.stack([convert(frame) for frame in frames])


