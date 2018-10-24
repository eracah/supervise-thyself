import torch
import sys
from models.utils import get_weights_path
from models.base_encoder import Encoder
from models.inverse_model import InverseModel
from models.forward_model import ForwardModel
from models.baselines import RawPixelsEncoder,RandomLinearProjection,RandomWeightCNN, VAE, BetaVAE
from evaluations.eval_models import EvalModel, ForwardEvalModel


def setup_model(args, env):
    model_table = {"inv_model":InverseModel, "vae":VAE, "raw_pixel": RawPixelsEncoder,
                                 "lin_proj": RandomLinearProjection,
                                 "rand_cnn": RandomWeightCNN,  "beta_vae": BetaVAE}

    encoder_kwargs = dict(in_ch=3,im_wh=args.resize_to,h_ch=args.hidden_width, embed_len=args.embed_len,  
                        num_actions=env.action_space.n, beta=args.beta)

    model_name = args.model_name.split("forward_")[-1] 
    base_model = model_table[model_name](**encoder_kwargs).to(args.device)
    encoder = base_model if model_name in ["lin_proj", "raw_pixel", "rand_cnn"] else base_model.encoder
    
    # train (not forward)
    if args.mode == "train":
        model = base_model
        
    # eval for forward_models
    elif args.mode == "eval" and "forward_" in args.model_name:
        model = setup_eval_forward_model(encoder,args,env)
    
    # train_forward, eval (not forward) and test
    else:
        this_module = sys.modules[__name__]
        setup_fn = getattr(this_module, "setup_" + args.mode + "_model")
        model = setup_fn(encoder,args,env)
        
    return model    




def setup_eval_forward_model(encoder,args,env):
    num_actions = env.action_space.n
    forward_model = ForwardModel(encoder, n_actions=num_actions).to(args.device)
    load_weights(forward_model, args)
    eval_forward_model = ForwardEvalModel(forward_predictor=forward_model,
                   num_classes=env.nclasses_table[args.label_name], args=args).to(args.device)
    #load_weights(eval_forward_model.forward_predictor,args)
    return eval_forward_model
    
    
    
def setup_test_model(encoder,args,env):
    eval_model = EvalModel(encoder=encoder,
                   num_classes=env.nclasses_table[args.label_name], args=args).to(args.device)
    load_weights(eval_model,args)
    return eval_model
    


def setup_eval_model(encoder,args, env):
    eval_model = EvalModel(encoder=encoder,
                   num_classes=env.nclasses_table[args.label_name], args=args).to(args.device)
    load_weights(eval_model.encoder,args)
    return eval_model
    


def setup_train_forward_model(encoder,args, env):
    num_actions = env.action_space.n
    load_weights(encoder, args)
    model = ForwardModel(encoder, n_actions=num_actions).to(args.device)
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
    

