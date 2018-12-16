import torch
import sys
from models.utils import get_weights_path
from models.base_encoder import Encoder
from models.inverse_model import InverseModel
from models.forward_model import ForwardModel
from models.seq_verif import ShuffleNLearn
from models.baselines import RawPixelsEncoder,RandomLinearProjection,RandomWeightCNN
from models.vae import VAE
from evaluations.inference_models import InferenceEvalModel
from evaluations.prediction_models import ForwardEvalModel



def setup_model(args):
    model_table = {"inv_model":InverseModel, "vae":VAE, "rand_cnn": RandomWeightCNN, "snl": ShuffleNLearn}
    
    encoder_kwargs = dict(in_ch=3,
                          im_wh=args.resize_to,
                          h_ch=args.hidden_width,
                          embed_len=args.embed_len,  
                          num_actions=args.num_actions,
                          base_enc_name=args.base_enc_name,
                         args=args)
    
    model_name = args.model_name.split("forward_")[-1] 
    
    base_model = model_table[model_name](**encoder_kwargs).to(args.device)
    
    encoder = base_model if model_name in ["lin_proj", "raw_pixel", "rand_cnn"] else base_model.encoder
    
    # train (not forward)
    if args.mode == "train":
        model = base_model
        
    # eval for forward_models
    elif args.mode == "eval" and "forward_" in args.model_name:
        model = setup_eval_forward_model(encoder,args)
        
    elif args.mode == "test" and "forward_" in args.model_name:
        model = setup_test_forward_model(encoder,args)
    
    # train_forward, eval (not forward) and test (not forward)
    else:
        this_module = sys.modules[__name__]
        setup_fn = getattr(this_module, "setup_" + args.mode + "_model")
        model = setup_fn(encoder,args)
        
    return model    

def setup_eval_ctl_model(encoder,args):
    from evaluations.control_models import ControlEvalModel
    eval_model = ControlEvalModel(encoder=encoder,
                   num_actions=args.num_actions, args=args).to(args.device)
    load_weights(eval_model.encoder,args)
    return eval_model
    

def setup_test_forward_model(encoder,args):
    forward_model = ForwardModel(encoder, n_actions=args.num_actions).to(args.device)
    eval_forward_model = ForwardEvalModel(forward_predictor=forward_model,
                   num_classes=args.nclasses_table[args.label_name], args=args).to(args.device)
    load_weights(eval_forward_model, args)
    return eval_forward_model
    

def setup_eval_forward_model(encoder,args):
    forward_model = ForwardModel(encoder, n_actions=args.num_actions).to(args.device)
    load_weights(forward_model, args)
    eval_forward_model = ForwardEvalModel(forward_predictor=forward_model,
                   num_classes=args.nclasses_table[args.label_name], args=args).to(args.device)
    #load_weights(eval_forward_model.forward_predictor,args)
    return eval_forward_model
    
    
    
def setup_test_model(encoder,args):
    eval_model = InferenceEvalModel(encoder=encoder,
                   num_classes=args.nclasses_table[args.label_name], args=args).to(args.device)
    load_weights(eval_model,args)
    return eval_model
    


def setup_eval_model(encoder,args):
    eval_model = InferenceEvalModel(encoder=encoder,
                   num_classes=args.nclasses_table[args.label_name], args=args).to(args.device)
    load_weights(eval_model.encoder,args)
    return eval_model
    


def setup_train_forward_model(encoder,args):
    load_weights(encoder, args)
    model = ForwardModel(encoder, n_actions=args.num_actions).to(args.device)
    args.model_name = "forward_" + args.model_name     
    # now it should behave exactly like train mode, so rename mode to train
    #args.mode = "train"
    return model
    
    


# In[38]:


def load_weights(model, args):
    """This function changes the state of model or args. They are mutable"""
    weights_path = get_weights_path(args)
    print(weights_path)
    if weights_path:
        model.load_state_dict(torch.load(str(weights_path)))
        args.loaded_weights = True
    else:
        print("No weights available for %s. Using randomly initialized %s"%(args.model_name,args.model_name))
        args.loaded_weights = False
    

