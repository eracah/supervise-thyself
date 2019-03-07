import torch
import sys
from models.base_encoder import Encoder
from models.inverse_model import InverseModel
from models.tdc import TDC
from models.shuffle_n_learn import ShuffleNLearn
from models.random_baselines import RawPixelsEncoder,RandomLinearProjection,RandomWeightCNN
from models.vae import VAE
from evaluations.infer_model import InferModel
from evaluations.predict_model import PredictModel
from pathlib import Path
from utils import get_child_dir
import copy
import numpy as np
from data.env_utils.env_setup import setup_env


def setup_model(args):
    model_table = {"inv_model":InverseModel, "vae":VAE,
                   "rand_cnn": RandomWeightCNN, "snl": ShuffleNLearn,
                   "tdc": TDC}
    
    
    env = setup_env(args)
    args.num_actions = env.action_space.n
    del env
    
    encoder_kwargs = dict(in_ch=3,
                          im_wh=args.resize_to,
                          h_ch=args.hidden_width,
                          embed_len=args.embed_len,  
                          num_actions=args.num_actions,
                          base_enc_name=args.base_enc_name,
                          args=args)
    
    
    embedder_name = args.embedder_name
    base_model = model_table[embedder_name](**encoder_kwargs).to(args.device)
    encoder = base_model if embedder_name in ["rand_cnn"] else base_model.encoder
    
    this_module = sys.modules[__name__]
    setup_fn = getattr(this_module, "setup_" + args.task + "_model")
    model = setup_fn(base_model, encoder, args)

    return model    



def setup_viz_model(base_model,encoder, args):
    load_weights(base_model.encoder, args)
    return base_model
    
    
def setup_embed_model(base_model, encoder, args):
    assert args.embedder_name != "rand_cnn", "Random CNN needs no training!"
    return base_model

def setup_infer_model(base_model,encoder,args):
    infer_model = InferModel(encoder=encoder,
                   num_classes=args.nclasses_table[args.label_name], args=args).to(args.device)
    if args.mode == "train": # linear classifier is randomly initialized here
        load_weights(infer_model.encoder, args)
    elif args.mode == "test": # linear classifier is loaded from saved weights
        load_weights(infer_model, args)
    return infer_model

def setup_predict_model(base_model,encoder,args):
    predict_model = PredictModel(encoder=encoder,args=args).to(args.device)
    if args.mode == "train": # forward model is randomly initialized here 
        load_weights(predict_model.encoder, args)
    elif args.mode == "test": # forward model is loaded from saved weights
        load_weights(predict_model, args)        
    return predict_model

def load_weights(model, args):
    """This function changes the state of model or args. They are mutable"""
    weights_path = get_weights_path(args)
    if weights_path:
        model.load_state_dict(torch.load(str(weights_path)))
        args.loaded_weights = True
        print("model load dir",weights_path)
    else:
        print("No weights available for %s. Using randomly initialized %s"%(args.embedder_name,args.embedder_name))
        args.loaded_weights = False


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
    print("looking for: ", base_path)


    #print(base_path)
    if not base_path.exists():
        return None

    for hyp_dir in base_path.iterdir():
        if args.mode != "viz":
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
    

