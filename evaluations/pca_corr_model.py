import torch
import copy
#from sklearn.metrics import explained_variance_score, r2_score
from sklearn.decomposition import PCA
from scipy.stats import spearmanr

def compute_pca_corr(embeddings,labels, num_components=5):
        pca = PCA(n_components=num_components)
        print(embeddings.shape)
        pcs = pca.fit_transform(embeddings.data)
        evr = pca.explained_variance_ratio_
        sp_corr = [spearmanr(labels,pcs[:,i])[0] for i in range(num_components)]
        inds = [str(i+1) for i in range(num_components)]
        evr_dict = dict(zip(inds,evr))
        sp_corr_dict = dict(zip(inds,sp_corr))
        
        return sp_corr_dict , evr_dict
        

        