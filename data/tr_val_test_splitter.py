from data.replay_buffer import BufferFiller
from utils import setup_env, convert_frame

def setup_tr_val_test(env, sizes, policy, convert_fxn,batch_size, frames_per_trans=2, just_train=False):
    bf = BufferFiller(convert_fxn=convert_fxn, env=env, policy=policy,
                  batch_size=batch_size) 
    datasets = [bf.fill(size) for size in sizes]
    return datasets
    



if __name__ == "__main__":
    seed = 10
    env, action_space, grid_size, num_directions,tot_examples, random_policy = setup_env(env_name="originalGame-v0",
                                                                                          seed = seed)



    tr,val, te = setup_tr_val_test(env,[600,10,5],policy=random_policy, convert_fxn=convert_frame,batch_size=2)

    tr.sample()