from data.replay_buffer import BufferFiller
from data.utils import convert_frame
import numpy as np

def setup_tr_val_test(env, args):
    if args.mode != "test":
        sizes = [args.tr_size,args.val_size]
    else:
        sizes = [args.test_size]

    bf = BufferFiller(env, args) 
    bufs = [bf.fill(size) for size in sizes]
    if args.resize_to[0] == -1:
        args.resize_to = bufs[0].memory[0].xs[0].shape[:2]
    return bufs
    



if __name__ == "__main__":
    from data.utils import setup_env, convert_frame
    seed = 10
    env, action_space, grid_size, num_directions,tot_examples, random_policy = setup_env(env_name="originalGame-v0",
                                                                                          seed = seed)



    tr,val, te = setup_tr_val_test(env,[600,10,5],policy=random_policy, convert_fxn=convert_frame,batch_size=2)

    tr.sample()