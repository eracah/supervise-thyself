from pathlib import Path
from utils import get_child_dir
import copy
import numpy as np

def get_weights_path(args):
    # load embedding task encoder weights if training and load from weights trained on embed_env
    if args.regime == "transfer":
        weights_task = "embed"
        weights_env_name = args.embed_env
        weights_level_name = args.embed_level
    # load current task model weights if testing and load from weights trained on transfer_env
    elif args.regime == "test":
        weights_task = args.task
        weights_env_name = args.transfer_env
        weights_level_name = args.transfer_level
        
    elif args.regime == "embed":
        assert False, "no need to load weights for embed!"


    best_loss = np.inf
    weights_path = None
    base_path = Path(".models") / get_child_dir(args,
                                                task=weights_task,
                                                env_name=weights_env_name,
                                                level=weights_level_name).parent


    #print(base_path)
    if not base_path.exists():
        return None

    for hyp_dir in base_path.iterdir():

        if (args.test_notebook and "nb" not in hyp_dir.name) or (not args.test_notebook and "nb" in hyp_dir.name):
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