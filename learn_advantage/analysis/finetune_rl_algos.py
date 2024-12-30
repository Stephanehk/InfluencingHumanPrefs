import pickle
import numpy as np
import gzip
import matplotlib.pyplot as plt
import torch
import gymnasium as gym
from gymnasium import spaces
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO,DQN
from stable_baselines3.common.env_checker import check_env

from learn_advantage.algorithms.rl_algos import (
    build_random_policy,
    get_gt_avg_return,
    iterative_policy_evaluation,
    build_pi_from_feats,
    q_learning,
    ppo,
    value_iteration,
    q_learning_timestep_checkpointed
)

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,trial, extended_SF, rew_vec=None):
        super().__init__()
        self.gt_rew_vec = np.load(gzip.GzipFile("/Users/stephanehatgiskessell/Documents/AAAI_Regret_Based_Reward_Learning/data/input/random_MDPs/MDP_" + str(trial) +"gt_rew_vec.npy.gz","r"))
        with open("/Users/stephanehatgiskessell/Documents/AAAI_Regret_Based_Reward_Learning/data/input/random_MDPs/MDP_" + str(trial) +"env.pickle", 'rb') as rf:
            self.env = pickle.load(rf)
            self.env.generate_transition_probs()
            self.env.set_custom_reward_function(self.gt_rew_vec)
        self.extended_SF = extended_SF

        if self.extended_SF:
            assert rew_vec is not None
            self.rew_vec = rew_vec
        else:
            self.rew_vec = self.gt_rew_vec


        self.action_space = spaces.Discrete(4)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(self.env.height*self.env.width,), dtype=np.uint8)
        self.state = self.env.get_start_state()
        self.n_timesteps =0
        self.max_timesteps = 1000


    def step(self, a_index):
        next_state, reward, terminated, _ = self.env.get_next_state(self.state, a_index)

        self.n_timesteps += 1
        if self.extended_SF:
            reward = self.rew_vec[self.state[0]][self.state[1]][a_index]
        reward = float(reward)

        observation = np.zeros((self.env.height, self.env.width), dtype=np.uint8)
        observation[next_state[0]][next_state[1]] = 1
        observation = observation.flatten()

        truncated = False
        if self.n_timesteps >  self.max_timesteps:
            truncated = True
            done = True
    
        self.state = next_state
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.n_timesteps =0
        self.state = self.env.get_start_state()

        observation = np.zeros((self.env.height, self.env.width), dtype=np.uint8)
        observation[self.state[0]][self.state[1]] = 1
        observation = observation.flatten()

        info = {}

        return observation, info

    def render(self):
        pass

    def close(self):
        pass

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, env,gt_rew_vec, save_header, verbose=0, update_freq=2048):
        super().__init__(verbose)
        self.env = env
        
        self.gt_rew_vec = gt_rew_vec
        self.save_header = save_header
        self.n_hits = 0
        self.update_freq = update_freq


        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        self.n_calls = 0  # type: int
        self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.avg_returns = []

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        
        """
        if self.n_calls % self.update_freq != 0:
            return

        self.n_hits += 1
        pi = {}
        actions = torch.tensor([i for i in range(4)])
        for i, j in self.env.positions():
            state_one_hot = torch.zeros((self.env.height, self.env.width))
            state_one_hot[i][j] = 1
            state_one_hot = state_one_hot.flatten().unsqueeze(0)
            log_probs = self.model.policy.get_distribution(state_one_hot).log_prob(actions).detach().numpy()
            probs = np.exp(log_probs)
            pi[(i,j)] = [(probs[a], a) for a in range(4)]

        V_under_gt = iterative_policy_evaluation(pi, rew_vec=self.gt_rew_vec, env=self.env)
        avg_return = np.sum(V_under_gt) / self.env.n_starts
        self.avg_returns.append(avg_return)

    def _init_callback(self) -> None:
        pass

    # def _on_training_start(self) -> None:
    #     """
    #     This method is called before the first rollout starts.
    #     """
    #     pass

    # def _on_rollout_start(self) -> None:
    #     """
    #     A rollout is the collection of environment interaction
    #     using the current policy.
    #     This event is triggered before collecting new samples.
    #     """
    #     pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        """
        return True
    
    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        np.save("test_reward_shapped_assum/" + self.save_header + ".npy",self.avg_returns)


class CustomCallback_Q(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, env,gt_rew_vec, save_header, verbose=0, update_freq=2048):
        super().__init__(verbose)
        self.env = env
        
        self.gt_rew_vec = gt_rew_vec
        self.save_header = save_header
        self.n_hits = 0
        self.update_freq = update_freq


        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        self.n_calls = 0  # type: int
        self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.avg_returns = []

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        
        """
        if self.n_calls % self.update_freq != 0:
            return

        self.n_hits += 1
        pi = {}
        actions = torch.tensor([i for i in range(4)])
        for i, j in self.env.positions():
            state_one_hot = torch.zeros((self.env.height, self.env.width))
            state_one_hot[i][j] = 1
            state_one_hot = state_one_hot.flatten().unsqueeze(0)
            pred_q = self.model.policy.q_net(state_one_hot).detach().numpy().tolist()[0]
            V = max(pred_q)
            V_count = pred_q.count(V)
            pi[(i, j)] = [
                (1 / V_count if pred_q[a_index] == V else 0, a_index)
                for a_index in range(4)
            ]

        V_under_gt = iterative_policy_evaluation(pi, rew_vec=self.gt_rew_vec, env=self.env)
        avg_return = np.sum(V_under_gt) / self.env.n_starts
        self.avg_returns.append(avg_return)

    def _init_callback(self) -> None:
        pass

    # def _on_training_start(self) -> None:
    #     """
    #     This method is called before the first rollout starts.
    #     """
    #     pass

    # def _on_rollout_start(self) -> None:
    #     """
    #     A rollout is the collection of environment interaction
    #     using the current policy.
    #     This event is triggered before collecting new samples.
    #     """
    #     pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        """
        return True
    
    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        np.save("test_reward_shapped_assum/" + self.save_header + ".npy",self.avg_returns)



mode = "deterministic"
num_prefs = 100000
model_ = "regret_pr"

update_freq = 16384
total_timesteps = update_freq*7

# total_timesteps = 38400*5

plot_interval = int((total_timesteps/update_freq)+1)

all_q_areas = []
all_ppo_areas = []
all_dqn_areas = []

def learn_policy(reward_type, q_learning_params, ppo_learning_params, dqn_learning_params):
    # print (ax)
   
    n_trials_used = 0
    r_a_hat_q_scaled_rets = []
    r_a_hat_ppo_scaled_rets = []
    r_a_hat_dqn_scaled_rets = []
   
    n_trials_used = 0

    all_r_hat_a_value_iteration_scaled_rets = np.load("/Users/stephanehatgiskessell/Documents/8:9:23_SAVE_ME_RESULTS/SAVE_ME_scaled_returns/pr_scaled_returns_regret_pr_mode="+mode+"_num_prefs="+str(num_prefs)+".npy")
    r_hat_a_value_iteration_scaled_rets = []

    for trial in range(100,130):
        # print ("ON TRIAL " + str(trial))
        n_trials_used += 1

        gt_rew_vec = np.load(gzip.GzipFile("/Users/stephanehatgiskessell/Documents/AAAI_Regret_Based_Reward_Learning/data/input/random_MDPs/MDP_" + str(trial) +"gt_rew_vec.npy.gz","r"))
        gt_rew_vec = np.array(gt_rew_vec)
        with open("/Users/stephanehatgiskessell/Documents/AAAI_Regret_Based_Reward_Learning/data/input/random_MDPs/MDP_" + str(trial) +"env.pickle", 'rb') as rf:
            env = pickle.load(rf)
            env.generate_transition_probs()
            env.set_custom_reward_function(gt_rew_vec)
        if reward_type == "approx_oaf":
            extended_SF = True
            fp = "/Users/stephanehatgiskessell/Documents/8:9:23_SAVE_ME_RESULTS/REDONE_SAMPLING_SAVE_ME_checkpointed_reward_vecs/"+str(trial)+"_True_"+model_+"_mode="+str(mode)+"_extended_SF=True_generalize_SF=False_num_prefs="+str(num_prefs)+"rew_vects.npy"
            oaf = np.load(fp)[-1]
            func = oaf.reshape((env.height, env.width, 4))
            linestyle = "solid"
            alg_desc = "A*_hat"
            ppo_color = "#800080"
            q_color = "#FF8C00"
            dqn_color ="#6B8E23"
        elif reward_type == "true_oaf":
            extended_SF = True
            V,Qs = value_iteration(rew_vec = gt_rew_vec,env=env)
            func = np.zeros((env.height, env.width, 4))
            for h in env.row_iter():
                for w in env.column_iter():
                    for a in range(4):
                        func[h][w][a] = Qs[h][w][a] - V[h][w]    
            linestyle = "dotted"
            alg_desc = "A*"
            ppo_color = "#9370DB"
            q_color = "#FFA500"
            dqn_color ="#9ACD32"
        elif reward_type == "true_rew":
            func = gt_rew_vec
            extended_SF = False  
            linestyle = "dashed"
            alg_desc = "r'"
            ppo_color = "#DDA0DD"
            q_color = "#FFDAB9"
            dqn_color = "#ADFF2F"
        else:
            print ("uncrecognized reward type")
            assert False

        random_avg_return = np.load("/Users/stephanehatgiskessell/Documents/legacy_reward_learning_stuff/Regret_Based_Reward_Learning_From_Desktop/Reward_Learning/test_reward_shapped_assum/"+str(trial)+"_random_avg_return.npy")
        gt_avg_return = np.load("/Users/stephanehatgiskessell/Documents/legacy_reward_learning_stuff/Regret_Based_Reward_Learning_From_Desktop/Reward_Learning/test_reward_shapped_assum/"+str(trial)+"_gt_avg_return.npy")
        oaf_scaled_return = np.load("/Users/stephanehatgiskessell/Documents/legacy_reward_learning_stuff/Regret_Based_Reward_Learning_From_Desktop/Reward_Learning/test_reward_shapped_assum/"+str(trial)+"_"+mode+"_"+str(num_prefs)+"oaf_scaled_return.npy")
        r_hat_a_value_iteration_scaled_rets.append(np.clip(all_r_hat_a_value_iteration_scaled_rets[trial-100],-1,1))


        if not os.path.exists("test_reward_shapped_assum/"):
            os.makedirs("test_reward_shapped_assum/")

        #==========================================================================================
        # save_header_q = str(trial)+"_"+mode+"_"+str(num_prefs)+"prefs_"+reward_type+"_avg_returns_q_learning"
        # q_hyperparam_header = "_epsilon=" + str(q_learning_params[0]) + "_decay_rate=" + str(q_learning_params[1])
        # #look at all timesteps:
        # _, r_a_hat_avg_return_q_learning = q_learning_timestep_checkpointed(rew_vec=func, alpha = 1 ,env=env,extended_SF=extended_SF,n_timesteps=total_timesteps, return_training_curve=False, gt_rew_vec=gt_rew_vec,checkpoint_every=update_freq, epsilon=q_learning_params[0], decay_rate=q_learning_params[1])
        # #look at only the last timestep:
        # # _, r_a_hat_avg_return_q_learning = q_learning_timestep_checkpointed(rew_vec=func, alpha = 1 ,env=env,extended_SF=extended_SF,n_timesteps=total_timesteps, return_training_curve=False, gt_rew_vec=gt_rew_vec,checkpoint_every=update_freq, epsilon=q_learning_params[0], decay_rate=q_learning_params[1])
        # np.save("test_reward_shapped_assum_finetuning/" + save_header_q +q_hyperparam_header+ ".npy", r_a_hat_avg_return_q_learning)

        # r_a_hat_q_scaled_return = (r_a_hat_avg_return_q_learning - random_avg_return)/(gt_avg_return - random_avg_return)
        # r_a_hat_q_scaled_return = np.clip(r_a_hat_q_scaled_return,-1,1)

        # r_a_hat_q_scaled_rets.append(r_a_hat_q_scaled_return)

        # hyperparam_header = q_hyperparam_header
        # scaled_rets = r_a_hat_q_scaled_rets
        #==========================================================================================
        # wrapped_env = CustomEnv(trial =trial, extended_SF=extended_SF,rew_vec=func)
        # save_header_ppo = str(trial)+"_"+mode+"_"+str(num_prefs)+"prefs_"+reward_type+"_avg_returns_ppo"  
        # ppo_hyperparam_header = "_lr=" + str(ppo_learning_params[0]) + "_gae_lambda=" + str(ppo_learning_params[1]) + "_normalize_adv=" + str(ppo_learning_params[2]) + "_layer_size=" + str(ppo_learning_params[3])
        # save_header_ppo += ppo_hyperparam_header
        
        # #look at all timesteps:
        # checkpoint_callback = CustomCallback(wrapped_env.env, wrapped_env.gt_rew_vec, save_header_ppo, update_freq=update_freq)
        # #look at only the last timestep:
        # # checkpoint_callback = CustomCallback(wrapped_env.env, wrapped_env.gt_rew_vec, save_header_ppo, update_freq=total_timesteps-32)

        # check_env(wrapped_env)
        # policy_kwargs = dict(activation_fn=torch.nn.Tanh,
        #                 net_arch=dict(pi=[ppo_learning_params[3]], vf=[ppo_learning_params[3]]))
        # model = PPO("MlpPolicy", wrapped_env, verbose=0,n_steps=32,batch_size=32,gae_lambda= ppo_learning_params[1], learning_rate=ppo_learning_params[0],normalize_advantage=ppo_learning_params[2],policy_kwargs=policy_kwargs)
        # model.learn(total_timesteps=total_timesteps,callback=checkpoint_callback)

        # r_a_hat_avg_return_ppo_learning = np.load("test_reward_shapped_assum/" + save_header_ppo + ".npy")[-1]
        # r_a_hat_ppo_scaled_return = (r_a_hat_avg_return_ppo_learning - random_avg_return)/(gt_avg_return - random_avg_return)
        # r_a_hat_ppo_scaled_return = np.clip(r_a_hat_ppo_scaled_return,-1,1)

        # r_a_hat_ppo_scaled_rets.append(r_a_hat_ppo_scaled_return)

        # hyperparam_header = ppo_hyperparam_header
        # scaled_rets = r_a_hat_ppo_scaled_rets
        
        #==========================================================================================
        wrapped_env = CustomEnv(trial =trial, extended_SF=extended_SF,rew_vec=func)
        save_header_dqn = str(trial)+"_"+mode+"_"+str(num_prefs)+"prefs_"+reward_type+"_avg_returns_dqn"
        dqn_hyperparam_header = "_lr=" + str(dqn_learning_params[0]) + "_learning_starts=" + str(dqn_learning_params[1]) + "_target_update_interval=" + str(dqn_learning_params[2]) + "_exp_frac=" + str(dqn_learning_params[3]) + "_layer_size=" + str(dqn_learning_params[4])
        save_header_dqn += dqn_hyperparam_header

        #look at all timesteps:
        checkpoint_callback = CustomCallback_Q(wrapped_env.env, wrapped_env.gt_rew_vec, save_header_dqn, update_freq=update_freq)
        #look at only the last timestep:
        # checkpoint_callback = CustomCallback_Q(wrapped_env.env, wrapped_env.gt_rew_vec, save_header_dqn, update_freq=total_timesteps-32)
        policy_kwargs = dict(activation_fn=torch.nn.Tanh,
                        net_arch=[dqn_learning_params[4]])
        model = DQN("MlpPolicy", wrapped_env,gamma=0.999,learning_rate=dqn_learning_params[0],learning_starts=dqn_learning_params[1],target_update_interval=dqn_learning_params[2],exploration_fraction=dqn_learning_params[3], verbose=0,policy_kwargs=policy_kwargs)
        model.learn(total_timesteps=total_timesteps,callback=checkpoint_callback)

        r_a_hat_avg_return_dqn_learning = np.load("test_reward_shapped_assum/" + save_header_dqn + ".npy")[-1]
        r_a_hat_dqn_scaled_return = (r_a_hat_avg_return_dqn_learning - random_avg_return)/(gt_avg_return - random_avg_return)
        r_a_hat_dqn_scaled_return = np.clip(r_a_hat_dqn_scaled_return,-1,1)

        r_a_hat_dqn_scaled_rets.append(r_a_hat_dqn_scaled_return)

        hyperparam_header = dqn_hyperparam_header
        scaled_rets = r_a_hat_dqn_scaled_rets

      
          
    print (reward_type)
    print (hyperparam_header)
    print (np.mean(scaled_rets))
    print ("\n")


# reward_type = "true_rew"
for reward_type in ["true_rew", "true_oaf", "approx_oaf"]:
    # learn_policy(reward_type, [0.8, 0.999], None, None)
    # learn_policy(reward_type, [0.8, 0.99], None, None)
    # learn_policy(reward_type, [0.8, 0.9], None, None)

    # learn_policy(reward_type, [0.4, 0.999], None, None)
    # learn_policy(reward_type, [0.4, 0.99], None, None)
    # learn_policy(reward_type, [0.4, 0.9], None, None)

    # learn_policy(reward_type, [0.2, 0.999], None, None)
    # learn_policy(reward_type, [0.2, 0.99], None, None)
    # learn_policy(reward_type, [0.2, 0.9], None, None)

    # winner



    # learn_policy(reward_type, None, [0.0003, 0.95, True, 64], None)
    # learn_policy(reward_type, None, [0.003, 0.95, True, 64], None)
    # learn_policy(reward_type, None, [0.03, 0.95, True, 64], None)
    # learn_policy(reward_type, None, [0.3, 0.95, True, 64], None)

    # learn_policy(reward_type, None, [0.0003, 0.99, True, 64], None)
    # learn_policy(reward_type, None, [0.0003, 0.90, True, 64], None)

    # learn_policy(reward_type, None, [0.0003, 0.95, False, 64], None)

    # learn_policy(reward_type, None, [0.0003, 0.95, True, 32], None)
    # learn_policy(reward_type, None, [0.0003, 0.95, True, 128], None)



    # learn_policy(reward_type, None, None, [0.0001, 100, 10000, 0.1, 64])
    # learn_policy(reward_type, None, None, [0.001, 100, 10000, 0.1, 64])
    # learn_policy(reward_type, None, None, [0.01, 100, 10000, 0.1, 64])
    # learn_policy(reward_type, None, None, [0.1, 100, 10000, 0.1, 64])

    # learn_policy(reward_type, None, None, [0.0001, 10000, 10000, 0.1, 64])
    # learn_policy(reward_type, None, None, [0.0001, 1000, 10000, 0.1, 64])
    # learn_policy(reward_type, None, None, [0.0001, 10, 10000, 0.1, 64])

    # learn_policy(reward_type, None, None, [0.0001, 100, 1250, 0.1, 64])
    # learn_policy(reward_type, None, None, [0.0001, 100, 2500, 0.1, 64])
    # learn_policy(reward_type, None, None, [0.0001, 100, 5000, 0.1, 64])

    # learn_policy(reward_type, None, None, [0.0001, 100, 10000, 0.3, 64])
    # learn_policy(reward_type, None, None, [0.0001, 100, 10000, 0.5, 64])
    # learn_policy(reward_type, None, None, [0.0001, 100, 10000, 0.7, 64])

    # learn_policy(reward_type, None, None, [0.0001, 100, 10000, 0.1, 32])
    # learn_policy(reward_type, None, None, [0.0001, 100, 10000, 0.1, 128])
    pass


        # ppo_hyperparam_header = "_lr=" + str(ppo_learning_params[0]) + "_gae_lambda=" + str(ppo_learning_params[1]) + "_normalize_adv=" + str(ppo_learning_params[2]) + "_layer_size=" + str(ppo_learning_params[3])
