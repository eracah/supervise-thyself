
# coding: utf-8

# In[1]:


# post process tensorboard logs/csv, etc
def plot(eval_dict,args):
    from matplotlib import pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    for i,k in enumerate(eval_dict.keys()):
        plt.figure(i)
        plt.title("%s postion inference accuracy for the %s environment"%(k,args.env_name))
        anames, avg_acc = list(zip(*list(eval_dict[k]["avg_acc"].items())))
        bnames, std_err = list(zip(*list(eval_dict[k]["std_err"].items())))
        assert anames == bnames
        x = np.arange(len(avg_acc))
        plt.xticks(x, bnames)
        plt.errorbar(x,avg_acc,yerr=std_err,fmt="o")
        plt.show()     
        

