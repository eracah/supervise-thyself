from data.replay_buffer import BufferFiller
from data.utils import convert_frame
import time
import numpy as np
from data.replay_buffer import multicore_fill
def setup_tr_val_test(args):
    if args.mode != "test":
        sizes = [args.tr_size,args.val_size]
    else:
        sizes = [args.test_size]
    t0 = time.time()
    bf = BufferFiller(args)
    bufs = [bf.fill(size) for size in sizes]
    #bufs = [multicore_fill(size,args) for size in sizes]
    print("time for loading was %f"%(time.time() - t0))
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