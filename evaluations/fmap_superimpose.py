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