import torch
import sys
from models.utils import get_weights_path
from models.base_encoder import Encoder
from models.inverse_model import InverseModel
from models.tdc import TDC
from models.shuffle_n_learn import ShuffleNLearn
from models.random_baselines import RawPixelsEncoder,RandomLinearProjection,RandomWeightCNN
from models.vae import VAE
from evaluations.infer_model import InferModel
from evaluations.predict_model import PredictModel



def setup_model(args):
    model_table = {"inv_model":InverseModel, "vae":VAE,
                   "rand_cnn": RandomWeightCNN, "snl": ShuffleNLearn,
                   "tdc": TDC}
    
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


def setup_embed_model(base_model, encoder, args):
    return base_model

def setup_infer_model(base_model,encoder,args):
    infer_model = InferModel(encoder=encoder,
                   num_classes=args.nclasses_table[args.label_name], args=args).to(args.device)
    load_weights(infer_model.encoder, args)
    return infer_model

def setup_predict_model(base_model,encoder,args):
    predict_model = PredictModel(encoder=encoder,args=args).to(args.device)
    load_weights(predict_model.encoder, args)
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
    

