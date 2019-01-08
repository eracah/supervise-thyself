from pathlib import Path
from utils import get_child_dir
import copy
import numpy as np

def get_weights_path(args, weight_mode=None):
    mode,task, embedder_name, test_notebook = args.mode,args.task, args.embedder_name, args.test_notebook

    weight_mode=None

    assert task != "embed", "No need to load weights for embed"

    if mode == "train":
        weight_mode = "embed"
    elif mode == "test":
        weight_mode = task


    best_loss = np.inf
    weights_path = None
    base_path = Path(".models") / get_child_dir(args,task=weight_mode).parent


    #print(base_path)
    if not base_path.exists():
        return None

    for hyp_dir in base_path.iterdir():

        if (test_notebook and "nb" not in hyp_dir.name) or (not test_notebook and "nb" in hyp_dir.name):
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