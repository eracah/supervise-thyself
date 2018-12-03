from evaluations.control_models import ControlEvalModel
import gym
from data.utils import convert_frame
from training.base_trainer import BaseTrainer
from functools import partial
import copy
import numpy as np

class ControlTrainer(BaseTrainer):
    def __init__(self, model, args, experiment):
        super(ControlTrainer, self).__init__(model, args, experiment)
        self.eval_best_freq = self.args.eval_best_freq
        self.es = self.setup()
        self.model = model


    def setup(self):
        # dummy_env = gym.make(sellf.args.env_name)
        # num_actions = dummy_env.action_space.n
        # dummy_c = ControlModel(encoder=self.model,
        #                        num_actions=num_actions)
        num_params = np.prod(self.model.fc.weight.size()) + np.prod(self.model.fc.bias.size())
        param0 = np.random.randn(num_params)
        es = cma.CMAEvolutionStrategy(param0, sigma0=1,inopts={"popsize":self.args.popsize}) #maximize
        return es
        
    
    def train(self, model_dir, tr_buf=None,val_buf=None):
        evaluate = partial(base_evaluate,encoder=self.model.encoder, args=self.args)
        prev_best = np.inf
        for epoch in range(self.max_epochs):
            params_set, fitnesses = self.es.ask_and_eval(evaluate)
            es.tell(params_set,fitnesses)
            best_params, best_fitness, _ = es.best.get()
            if best_fitness < prev_best:
                best_ctlr = ControlEvalModel(best_overall_params)
                self.save_model(best_ctlr, 
                                model_dir,
                                "best_model_%f.pt"% -best_overall_fitness )
                prev_best = copy.deepcopy(best_fitness)
                
            best, worst, mean = np.min(fitnesses),\
                                np.max(fitnesses),\
                                np.mean(fitnesses)
            
            self.experiment.log_metrics(dict(best=-best,
                                             worst=-worst,
                                             mean=-mean),
                                        prefix="train",
                                        step=self.epoch)

            if epoch % eval_best_freq == 0:
                best_avg, best_dist = evaluate(parameters=best_overall_params,
                                                         rollouts=self.args.val_rollouts,
                                                        negative_reward=False,dist=True)
                self.experiment.log_metric(-best_avg,"val_best", step=self.epoch)
                
                
                
def base_evaluate(parameters,args, encoder,
                  negative_reward=True, dist=False):
    ctlr = ControlEvalModel(encoder=encoder,
                        num_actions=num_actions,
                        parameters=parameters)
    solution_rewards = []
    for _ in range(args.rollouts):
        reward_sum = do_rollout(ctlr=ctlr,
                                encoder=encoder,
                                args=args)
        if negative_reward: # for cases where the cma-es library minimizes
            reward_sum = - reward_sum
        solution_rewards.append(reward_sum) 
    avg_rew = np.mean(solution_rewards)
    if dist:
        return avg_rew, np.asarray(solution_rewards)
    else:
        return avg_rew

def do_rollout(ctlr,args):
    env = gym.make(args.env_name)
    done= False
    reward_sum = 0.
    state = env.reset()
    _ = env.render("rgb_array")     # render must come after reset
    while not done:
        x = convert_frame(state,to_tensor=True,
                              resize_to=args.resize_to)
        
        a = ctlr(x)
        state,reward,done,_ = env.step(a.data)
        reward_sum += reward
    return reward_sum