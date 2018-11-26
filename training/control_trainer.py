from evaluations.control_models import ControlModel
import gym
from data.utils import convert_frame

def evaluate(parameters, rollouts=1, negative_reward=True,dist=False):
    ctlr = ControlModel(embed_len=embed_len, num_actions=num_actions, parameters=parameters)
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

def do_rollout(ctlr, env_name, encoder,args):
    env = gym.make(env_name)
    done= False
    reward_sum = 0.
    state = env.reset()
    _ = env.render("rgb_array")     # render must come after reset
    while not done:
        state = convert_frame(state,to_tensor=True,resize_to=args.resize_to)
        z = encoder(state)
        a = ctlr(z)
        state,reward,done,_ = env.step(a.data)
        reward_sum += reward
    return reward_sum

class ControlTrainer(object):
    def __init__(self, model, args, experiment):
        self.model = model
        self.args = args
        self.model_name = self.args.model_name
        self.experiment = experiment
        self.eval_best_freq = self.args.eval_best_freq
        num_actions = env.action_space.n
        dummy_env = gym.make(args.env_name)
        num_actions = dummy_env.action_space.n
        dummy_c = ControlModel(embed_len=self.model.embed_len,num_actions=num_actions)
        num_params = np.prod(dummy_c.fc.weight.size()) + np.prod(dummy_c.fc.bias.size())
        param0 = np.random.randn(num_params)
        popsize = self.args.popsize
        self.es = cma.CMAEvolutionStrategy(param0, sigma0=1,inopts={"popsize":popsize}) #maximize
        self.generations = self.args.generations

    def train(self, model_dir):
        prev_best_fitness = np.inf
        for generation in range(self.generations):
            params_set, fitnesses = self.es.ask_and_eval(evaluate)
            es.tell(params_set,fitnesses)
            best_overall_params, best_overall_fitness, _ = es.best.get()
            if best_overall_fitness < prev_best_fitness:
                best_ctlr = ControlModel(best_overall_params)
            best_perf, worst_perf, pop_mean = np.min(fitnesses), np.max(fitnesses), np.mean(fitnesses)
            print(best_overall_fitness)

            if generation % eval_best_freq == 0:
                best_agent_avg, best_agent_dist = evaluate(best_overall_params,rollouts=1,
                                                           negative_reward=False,dist=True)