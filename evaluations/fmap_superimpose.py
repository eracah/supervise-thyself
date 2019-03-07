from matplotlib import pyplot as plt
from torchvision.utils import make_grid
import numpy as np
from torch import nn


def superimpose_fmaps(encoder, test_set, experiment):
    for i,trans in enumerate(test_set):
        xs = trans.xs
        xshape = list(tuple(xs.shape))
        xp = tuple([xshape[0] * xshape[1],*xshape[2:]])
        xs = xs.reshape(xp)
        fmap_model = list(encoder.children())[0]
        fmap = fmap_model(xs)
        ind = np.random.choice(xs.shape[0])
        
        fm_upsample = nn.functional.interpolate(fmap[ind,:,None],size=xs.shape[-2:],mode="bilinear")
        fm_grid = make_grid(fm_upsample,nrow=16,padding=0)[0].detach().numpy()
        xgrid = (xs[ind].repeat(1,16,16).numpy().transpose(1,2,0) + 1) /2
        

        fig = plt.figure(i,frameon=False,figsize=(50,50))
        im1 = plt.imshow(xgrid)
        im2 = plt.imshow(fm_grid, cmap=plt.cm.jet, alpha=0.7)

        experiment.log_figure(figure_name="it%i_ind%i"%(i,ind),figure=fig)
        if i > 8:
            break
            
            

def superimpose_seq_frames(encoder, model_name, test_set, fmap_index=0, ims_dir=None):

    trans = next(test_set.__iter__())

    # grab every third frame
    x_example = trans.xs[0][1::3]
    num_frames = x_example.shape[0]

    if encoder is not None:
        fmap_model = list(encoder.children())[0]
        fmap = fmap_model(x_example)
        fmap_pad = fmap[:,fmap_index,None]
        fmap_upsampled = nn.functional.interpolate(fmap_pad,size=x_example.shape[-2:],mode="bilinear",align_corners=True)
        fm_grid = make_grid(fmap_upsampled,nrow=num_frames,padding=0)[0].detach().numpy()
    
    xgrid = (make_grid(x_example,nrow=num_frames,padding=0).detach().numpy().transpose(1,2,0) + 1) /2
    
    #for i in range(1,10): #0.3 or 0.4
    #i = 5
    alpha=0.35
    fig = plt.figure(0,frameon=False,figsize=(50,50))
    plt.clf()
    plt.axis("off")
    im0 = plt.imshow(xgrid)
    plt.title("Raw Frames",fontdict={"size":40})
    fig = plt.figure(fmap_index,frameon=False,figsize=(50,50))
    plt.clf()
    plt.axis("off")
    im1 = plt.imshow(xgrid)

    if encoder is not None:
        im2 = plt.imshow(fm_grid, cmap=plt.cm.jet, alpha=alpha)
    plt.title("%s, feature map %i"%(model_name,fmap_index),fontdict={"size":40})
    
    if ims_dir is not None:
        plt.savefig(ims_dir / str("fmap%i.png"%(fmap_index)))
    #experiment.log_figure("fmap%i.png"%fmap_index,figure=fig)
