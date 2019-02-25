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
            
            

def superimpose_seq_frames(encoder, test_set, fmap_index):

        for i,trans in enumerate(test_set):#= next(test_set.__iter__())

            x_example = trans.xs[1][:-1:3]

            fmap_model = list(encoder.children())[0]

            fmap = fmap_model(x_example)

            fmap_i = nn.functional.interpolate(fmap[:,fmap_index,None],size=x_example.shape[-2:],mode="bilinear")

            fm_grid = make_grid(fmap_i,nrow=10,padding=0)[0].detach().numpy()

            #plt.imshow(fm_grid)

            xgrid = make_grid(x_example,nrow=10,padding=0)[0].detach().numpy()

            fig = plt.figure(i,frameon=False,figsize=(50,50))
            plt.clf()
            im1 = plt.imshow(xgrid)
            im2 = plt.imshow(fm_grid, cmap=plt.cm.viridis, alpha=0.7)
            #experiment.log_figure(figure_name="it%i_ind%i"%(i,ind),figure=fig)
            if i > 2:
                break