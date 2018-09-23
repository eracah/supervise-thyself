from data.replay_buffer import BufferFiller
from utils import setup_env, convert_frame

def setup_tr_val_val_test(env, policy, convert_fxn, tot_examples, batch_size, frames_per_trans, just_train=False, verbose=True):

    
    bf = BufferFiller(convert_fxn=convert_fxn, env=env, policy=policy,
                      batch_size=batch_size)


    frames_per_trans = 3
    tr_val_prop = 0.7
    num_tr_val = int((0.7*tot_examples) / frames_per_trans)
    tr_val_rb = bf.fill(num_tr_val, frames_per_trans=frames_per_trans)
    tr, val = bf.split(tr_val_rb,0.8)
    props = [0.1,0.1,0.1]
    names = ["eval_tr","eval_val", "test"]
    bufs = [tr, val]
    if verbose:
        print(len(tr))
        print(len(val))
    if not just_train:
        visited_buffer = tr + val
        for name, prop in zip(names,props):
            print(name)
            if verbose:
                print("creating %s_buf"%name)

            size = int(prop*tot_examples)

            buf = bf.fill_with_unvisited_states(visited_buffer=visited_buffer, size=size)
            if verbose:
                print(len(buf))
            assert size == len(buf), "%i, %i"%(size, len(buf))
            bufs.append(buf)
            visited_buffer += buf
    return bufs

 



if __name__ == "__main__":
    seed = 10
    env, action_space, grid_size, num_directions,tot_examples, random_policy = setup_env(env_name="MiniGrid-Empty-16x16-v0",
                                                                                          seed = seed)

    convert_fxn = convert_frame
    batch_size = 8 
    frames_per_trans =3 

    setup_tr_val_val_test(env, random_policy, convert_fxn, tot_examples, batch_size, frames_per_trans)