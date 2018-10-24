from pathlib import Path
from utils import get_child_dir
import copy
import numpy as np

def get_weights_path(args):
    if args.mode == "eval" and "forward_" not in args.model_name or args.mode == "train_forward":
        weight_mode = "train"
    if args.mode == "eval" and "forward_" in args.model_name:
        weight_mode = "train_forward"
    if args.mode == "test":
        weight_mode = "eval"
    best_loss = np.inf
    weights_path = None
    base_path = Path(".models") / get_child_dir(args,mode=weight_mode).parent
    #print(base_path)
    if not base_path.exists():
        return None
    for hyp_dir in base_path.iterdir():
        #print(hyp_dir)
        if args.test_notebook:
            if "nb" not in hyp_dir.name:
                continue
        else:
            if "nb" in hyp_dir.name:
                continue
        for model_dir in hyp_dir.iterdir():
            best_models = list(model_dir.glob("best_model*"))
            if len(best_models) == 0:
                continue
            model_path = best_models[0]
            #print(model_path)
            loss = float(str(model_path).split("/")[-1].split("_")[-1].split(".pt")[0])
            if loss < best_loss:
                best_loss = copy.deepcopy(loss)
                weights_path = copy.deepcopy(model_path)
            
    return weights_path