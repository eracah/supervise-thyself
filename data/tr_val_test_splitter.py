
# coding: utf-8

# In[1]:


from data.replay_buffer import BufferFiller

def setup_tr_val_val_test(env, policy, convert_fxn, tot_examples, batch_size, just_train=False, verbose=True):

    
    
    bf = BufferFiller(convert_fxn=convert_fxn, env=env, policy=policy,
                      batch_size=batch_size)
    
    props = [0.6,0.1,0.1,0.1,0.1]
    names = ["tr", "val", "eval_tr","eval_val", "test"]
    bufs = []
    for name, prop in zip(names,props):
        if verbose:
            print("creating %s_buf"%name)
        
        size = int(prop*tot_examples)
        if len(bufs) == 0:
            buf = bf.fill(size=size)
        else:
            visited_buffer = bufs[0]
            for b in bufs[1:]:
                visited_buffer = visited_buffer + b
                
            buf = bf.fill_with_unvisited_states(visited_buffer=visited_buffer, size=size)
        if verbose:
            print(len(buf))
        assert size == len(buf)
        bufs.append(buf)
        
        if just_train and name == "val":
            return bufs
        
    return bufs
    
 

