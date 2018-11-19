class C(nn.Module):
    def __init__(self,parameters=None):
        super(C,self).__init__()
        self.nz = args.nz
        self.nh = args.nh
        self.action_len = args.action_len
        if args.zandh:
            in_features = self.nz + self.nh
        else:
            in_features = self.nz
        self.fc = nn.Linear(in_features=in_features,out_features=self.action_len)
        if parameters is not None:
            weight_len = np.prod(self.fc.weight.size())
            weight = parameters[:weight_len]
            weight = np.resize(weight,self.fc.weight.size())
            bias = parameters[weight_len:]
            bias = np.resize(bias,self.fc.bias.size())
            self.fc.weight.set_(Variable(torch.from_numpy(weight).float()))
            self.fc.bias.set_(Variable(torch.from_numpy(bias).float()))
    
    def posprocess_output(self,raw_output):
        raw_steer, raw_gas, raw_brake = raw_output[0],raw_output[1],raw_output[2]

        steer = F.tanh(raw_steer) # between -1 and 1

        gas = F.softplus(raw_gas) # between 0 and 1

        brake = F.softplus(raw_brake) # between 0 and 1
        return steer,gas,brake
        
    
    def forward(self,z,h=None):
        
        z = z.squeeze()
        if args.zandh:
            assert (h is not None), "You must specify h, bro"
            h = h.squeeze()
            zh = torch.cat((z,h),dim=-1)
        else:
            zh = z

        raw_output = self.fc(zh)
        steer, gas, brake = self.posprocess_output(raw_output)
        action = torch.cat((steer,gas,brake))
        return action
    

def evaluate(parameters, rollouts=args.routs, negative_reward=True,dist=False):
    ctlr = C(parameters=parameters).cuda()
    solution_rewards = []
    for _ in range(rollouts):
        reward_sum = do_rollout(ctlr=ctlr)
        if negative_reward: # for cases where the cma-es library minimizes
            reward_sum = - reward_sum
        solution_rewards.append(reward_sum) 
    avg_rew = np.mean(solution_rewards)
    if dist:
        return avg_rew, np.asarray(solution_rewards)
    else:
        return avg_rew 
    
if __name__ == "__main__":
    env_name="CarRacing-v0"
    m = M(batch_size=1).cuda()
    m.load_state_dict(torch.load(args.mdn_rnn_weights))
    m.eval()
    
    
    V = VAE().cuda()
    V.load_state_dict(torch.load(args.vae_weights))
    V.eval()
    
    env = gym.make(env_name)

    

    dummy_c = C()
    num_params = np.prod(dummy_c.fc.weight.size()) + np.prod(dummy_c.fc.bias.size())
    param0 = np.random.randn(num_params)
    #c= C(param0).cuda()

    es = cma.CMAEvolutionStrategy(param0, 1,inopts={"popsize":args.popsize}) #maximize
    
    prev_best_fitness = np.inf
    for generation in range(args.generations):
        params_set, fitnesses = es.ask_and_eval(evaluate)
        es.tell(params_set,fitnesses)
        best_overall_params, best_overall_fitness, _ = es.best.get()
        if best_overall_fitness < prev_best_fitness:
            best_ctlr = C(best_overall_params).cuda()
            torch.save(best_ctlr.state_dict(),'%s/best_ctlr.pth' % (saved_model_dir))
            np.savez('%s/best_ctlr_params.npz' % (saved_model_dir),params=best_overall_params)
            prev_best_fitness = best_overall_fitness
        best_perf, worst_perf, pop_mean = np.min(fitnesses), np.max(fitnesses), np.mean(fitnesses)
        writer.add_scalar("best_performer",-best_perf,global_step=generation)
        writer.add_scalar("worst_performer",-worst_perf,global_step=generation)
        writer.add_scalar("pop_mean",-pop_mean,global_step=generation)
        
        if generation % args.eval_best_freq == 0:
            best_agent_avg, best_agent_dist = evaluate(best_overall_params,rollouts=args.rollouts_for_final_eval,
                                                       negative_reward=False,dist=True)
            
            writer.add_scalar("best_agent", best_agent_avg, global_step=generation)
            writer.add_histogram("best_agent_rew_dist", best_agent_dist, global_step=generation)
            
        

        