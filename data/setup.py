from data.env_utils import setup_env
from torch.utils.data.dataset import random_split
from data.datasets import EpisodeDataset, EnvDataset
from torch.utils.data import DataLoader
import numpy as np
import torch
from utils import setup_args
import sys

def setup_dataset(total_frames, args):
    env = setup_env(args.env_name)
    ds = EnvDataset(env=env,
                    total_frames=total_frames,
                    max_frames=args.episode_max_frames,
                    resize_to=args.resize_to,
                    frames_per_example=args.frames_per_example,
                    stride=args.stride)
    return ds

def setup_train_data(args):
    total_frames = args.tr_size + args.val_size
    ds = setup_dataset(total_frames, args)
    lens = [args.tr_size, args.val_size]
    print(lens)
    torch.manual_seed(0)
    tr,val= random_split(ds,lens)
    return tr,val

def setup_test_data(args):
    total_frames = args.test_size
    test_ds = setup_dataset(total_frames, args)
    return test_ds,
    
    

def setup_viz_data(args):
    return setup_test_data(args)

def setup_data(args):
    this_module = sys.modules[__name__]
    setup_dataset_fn = getattr(this_module, "setup_" + args.mode + "_data")
    datasets = setup_dataset_fn(args)
    
    dataloader_kwargs = dict(batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dataloaders = [DataLoader(dataset,**dataloader_kwargs) for dataset in datasets]
    return dataloaders