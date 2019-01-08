import torch
import copy
#from sklearn.metrics import explained_variance_score, r2_score
from sklearn.decomposition import PCA
from scipy.stats import spearmanr

class PCACorr(object):
    def __init__(self,encoder, sampler,num_components=4):
        self.encoder = encoder
        self.sampler = sampler
        self.num_components = num_components
    def collect_megabatch(self):
        fs = []
        spd = {k:[] for k in self.sampler.episodes[0].state_param_dict.keys()}
        for trans in self.sampler:
            x1 = trans.xs[:,0]
            f = self.encoder(x1)
            fs.append(f)
            for k,v in trans.state_param_dict.items():
                spd[k].append(copy.deepcopy(v[:,0]))

        f = torch.cat(fs)
    
        for k,v in trans.state_param_dict.items():
            spd[k] = torch.cat(spd[k]).squeeze()
        return f,spd
    
    def calc_pcs(self,f):
        pca = PCA()
        pcs = pca.fit_transform(f.data)
        return pcs, pca.explained_variance_ratio_
        
    def run(self):
        f, spd = self.collect_megabatch()
        pcs,evr = self.calc_pcs(f)
        sp_corr = self.calc_spearman_corr(pcs,spd)
        return sp_corr,evr[0]
    
    def calc_spearman_corr(self,pcs,spd):
        sp_corr = {}
        for k,v in spd.items():
            sp_corr[k] = spearmanr(v,pcs[:,0])[0]
        return sp_corr
        

        