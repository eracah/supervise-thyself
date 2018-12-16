import torch
import copy
from sklearn.metrics import explained_variance_score, r2_score
from sklearn.decomposition import PCA

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
        r2d = self.calc_r2(pcs,spd)
        return r2d,evr[0]
        
    def calc_r2(self,pcs,spd):
        r2d = {}
        for k,v in spd.items():
            r2d[k] = r2_score(v,pcs[:,0])
            
        return r2d
        