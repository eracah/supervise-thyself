from data.fillers import SamplerFiller, multicore_fill
from data.utils import convert_frame
import time
import numpy as np


def setup_tr_val_test(args):
    if args.mode == "test" or args.mode =="viz":
        sizes = [args.test_size]
    else:
        sizes = [args.tr_size,args.val_size]
    t0 = time.time()
    sf = SamplerFiller(args)
    samplers = [sf.fill(size) for size in sizes]
    #bufs = [multicore_fill(size,args) for size in sizes]
    print("time for loading was %f"%(time.time() - t0))
#     if args.resize_to[0] == -1:
#         args.resize_to = samplers[0].episodes[0].xs[0].shape[:2]
    return samplers
