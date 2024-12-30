import pickle
import os
import argparse
import gzip
import numpy as np
import torch
import random

from learn_advantage.utils.pref_dataset_utils import (
    augment_data,
)
from learn_advantage.utils.argparse_utils import parse_args, ArgsType
from learn_advantage.algorithms.advantage_learning import train
from learn_advantage.algorithms.rl_algos import (
    value_iteration,
    build_pi,
    iterative_policy_evaluation,
)
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

args = parse_args(ArgsType.DEFAULT)


force_cpu = args.force_cpu
keep_ties = args.keep_ties
n_prob_samples = args.n_prob_samples
n_prob_iters = args.n_prob_iters


def main():


    preference_model = args.preference_model  # how we generate prefs
    preference_assum = args.preference_assum  # how we learn prefs

  
    args.succ_feats = None
    args.succ_q_feats = None
    args.pis = None
    args.use_extended_SF = False
    args.learn_oaf = False

    if args.preference_assum == "regret":
        args.succ_feats = np.load("data/delivery_mdp/succ_feats_no_gt.npy", allow_pickle=True)
        args.pis = np.load("data/delivery_mdp/pis_no_gt.npy", allow_pickle=True)
        args.succ_q_feats = None
    else:
        args.succ_feats = args.succ_feats
        args.pis = None
        args.succ_q_feats = args.succ_q_feats

    if args.force_cpu:
        args.device = "cpu"
    else:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.preference_condition != "":
        args.preference_condition += "_"


    for seed in [100]:
        gt_rew_vec = np.array([-1,50,-50,1,-1,-2])
        with open("data/delivery_mdp/delivery_env.pickle", "rb") as rf:
            env = pickle.load(rf)
            env.generate_transition_probs()
            env.set_custom_reward_function(gt_rew_vec)
        
        #----------------------------------------------------------------------
        # if args.preference_assum == "regret":
        #     X = np.load("data/delivery_mdp/human_data/"+args.preference_condition+"DELIVERY_MDP_segment_pair_features_regret_form.npy")
        # elif args.preference_assum == "pr":
        #     X = np.load("data/delivery_mdp/human_data/"+args.preference_condition+"DELIVERY_MDP_segment_pair_features_pr_form.npy")
        # y = np.load("data/delivery_mdp/human_data/"+args.preference_condition+"DELIVERY_MDP_human_prefs.npy")
        # regret_X = np.load("data/delivery_mdp/human_data/"+args.preference_condition+"DELIVERY_MDP_segment_pair_features_regret_form.npy")
        #----------------------------------------------------------------------
        if args.preference_assum == "regret":
            X = np.load("data/delivery_mdp/human_data/different_start_state_DELIVERY_MDP_segment_pair_features_regret_form.npy")
        elif args.preference_assum == "pr":
            X = np.load("data/delivery_mdp/human_data/different_start_state_DELIVERY_MDP_segment_pair_features_pr_form.npy")

        if args.preference_model == "regret":
            y = np.load("data/delivery_mdp/human_data/different_start_state_DELIVERY_MDP_human_prefs_sigmoid_regret.npy")
        elif args.preference_model == "pr":
            y = np.load("data/delivery_mdp/human_data/different_start_state_DELIVERY_MDP_human_prefs_sigmoid_pr.npy")

        #----------------------------------------------------------------------

        # print (y.shape)
        # print (X.shape)
        #--------------------------
        # #REMOVE ALL SEGEMENT PAIRS WHERE THE G.T. REGRET MODEL IS INDIFFERENT BETWEEN SEGMENTS
        # env.set_custom_reward_function(gt_rew_vec)
        # V,_ = value_iteration(rew_vec =gt_rew_vec,env=env)
        # env.set_custom_reward_function(gt_rew_vec)

        # regret_X_ = regret_X.copy()
        # y_ = y.copy()

        # regret_X = []
        # y = []
        # from learn_advantage.utils.utils import sigmoid


        # for pref, x in zip(y_, regret_X_):
        #     regrets = []
        #     value_diff = []
        #     pr_diff = []
        #     for i in range(2):
        #         pr = np.dot(gt_rew_vec, x[i][:6])
        #         v_s0 = V[int(x[i][6])][int(x[i][7])]
        #         v_st = V[int(x[i][8])][int(x[i][9])]

        #         regrets.append((pr + v_st)-v_s0)
        #         value_diff.append(v_st-v_s0)
        #         pr_diff.append(pr)

        #     # if regrets[1] == regrets[0]:
        #     #     gt_pref = 0.5
        #     # elif regrets[1] > regrets[0]:
        #     #     gt_pref = 0
        #     # elif regrets[1] < regrets[0]:
        #     #     gt_pref = 1


        #     # r1_prob = sigmoid((regrets[1] - regrets[0]) / 1)
        #     # r2_prob = sigmoid((regrets[0] - regrets[1]) / 1)
        #     # num_regret = np.random.choice([1, 0], p=[r1_prob, r2_prob])

        #     r1_prob = sigmoid((pr_diff[1] - pr_diff[0]) / 1)
        #     r2_prob = sigmoid((pr_diff[0] - pr_diff[1]) / 1)
        #     num_pr = np.random.choice([1, 0], p=[r1_prob, r2_prob])
            
        #     #note: regret is a typo here, this is actually advantage
        #     # if regrets[0] == regrets[1]:
        #     #     num_pr = 0.5
        #     # elif regrets[0] > regrets[1]:
        #     #     num_pr = 0
        #     # elif regrets[0] < regrets[1]:
        #     #     num_pr = 1

        #     # if pr_diff[0] == pr_diff[1]:
        #     #     num_pr = 0.5
        #     # elif pr_diff[0] > pr_diff[1]:
        #     #     num_pr = 0
        #     # elif pr_diff[0] < pr_diff[1]:
        #     #     num_pr = 1
           
        #     # #TODO: maybe got these numbers switched
        #     # if num_regret == 0:
        #     #     #pref = [1, 0]
        #     #     gt_pref = 0
        #     # elif num_regret == 1:
        #     #     #pref = [0, 1]
        #     #     gt_pref = 1

        #     # regret_X.append(x)
        #     # print (regrets)
        #     # print (pr_diff)
        #     # print (num_regret)
        #     # print (num_pr)
        #     # print ("\n")
        #     y.append(num_pr)
        #     # y.append(pref)

        #     # if regrets[1] != regrets[0]:
        #     #     regret_X.append(x)
        #     #     y.append(pref)
        # # print (len(regret_X))
        # #-------------------------------------
        # n_incorrect = 0
        # n_total = 0
        # for pref1, pref2 in zip(y, y_):
        #     print (pref1)
        #     print (pref2)
        #     print ("\n")
        #     if pref1 != pref2:
        #         n_incorrect +=1
        #     n_total +=1
        # print (str(n_incorrect) + "/" + str(n_total))
        # assert False

        min_full_pref_size = 1812
        # for n in [1,2,5,10,20,50,100]: #number of partitions to split human data

        for n in (3,10,30,100):
            combined = list(zip(X, y))
            random.Random(seed).shuffle(combined)
            X_copy, y_copy = zip(*combined)

            partition_size = int(min_full_pref_size/n)
            partitioned_X = []
            partitioned_y = []
            for _ in range(n):
                partitioned_X.append(X_copy[:partition_size])
                partitioned_y.append(y_copy[:partition_size])
                if _ < n-1:
                    combined = list(zip(X_copy[partition_size:], y_copy[partition_size:]))
                    random.Random(seed).shuffle(combined)
                    X_copy, y_copy = zip(*combined)

            # partitioned_X = [X_copy[i::n] for i in range(n)]
            # partitioned_y = [y_copy[i::n] for i in range(n)]
            # regret_X_copys = [regret_X_copy[i::n] for i in range(n)]
        

            avg_returns = []
            num_near_opt = 0
            for partition_i, training_x in enumerate(partitioned_X):

                training_y = partitioned_y[partition_i]

                aX, ay = augment_data(training_x,training_y,"scalar")

                #shuffle_train_data=False for reproducibility
                rew_vect,_,_,_ = train(aX=aX, ay=ay, args=args, plot_loss=False,shuffle_train_data=False)
                print (rew_vect)
                #derive policy from learned reward function
                env.set_custom_reward_function(gt_rew_vec)
                _,Q = value_iteration(rew_vec =rew_vect,env=env)
                pi = build_pi(Q,env=env)

                V_under_gt = iterative_policy_evaluation(pi,env=env,rew_vec=gt_rew_vec)
                avg_return = np.sum(V_under_gt)/92 #number of possible start states
                print ("average return following learned policy: ")
                print (avg_return)
                print ("====================================================\n")
                if avg_return > 30:
                    num_near_opt+=1
                avg_returns.append(avg_return)

            print ("% near optimal: " + str(num_near_opt/n))
            #same_start_state_det_
            np.save("data/results/testing_things/different_start_state_stoch_" + str(seed) + args.preference_condition + preference_model + "_"+preference_assum+"_logistc_lin_partitioned" + "_avg_returns_n_split=" + str(n) + ".npy",avg_returns)
if __name__ == "__main__":
    main()