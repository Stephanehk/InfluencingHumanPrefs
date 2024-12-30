import numpy as np

import os
from prescriptive_effort.utils.argparse_utils import parse_args,ArgsType
from prescriptive_effort.analysis.prescriptive_estimator import Estimator



import torch
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import pickle

from learn_advantage.utils.segment_feats_utils import get_extended_features, format_y, format_X_pr, format_X_regret,format_X_adv, format_X_expected_return
from prescriptive_effort.utils.pref_dataset_utils import prefrence_pred_loss, format_regret_feats
from prescriptive_effort.analysis.models import LogisticRegression
from learn_advantage.utils.pref_dataset_utils import augment_data
from learn_advantage.algorithms.advantage_learning import train
from learn_advantage.algorithms.rl_algos import (
    value_iteration,
    build_random_policy,
    build_pi,
    get_gt_avg_return,
    iterative_policy_evaluation
)

args = parse_args(ArgsType.REWARD_LEARNING)
args.use_extended_SF = False
args.generalize_SF = False
args.include_actions = False
# args.use_synth_prefs = True


if args.force_cpu:
    args.device = torch.device("cpu")
else:
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gt_rew_vec = np.array([-1,50,-50,1,-1,-2])
with open("data/delivery_mdp/delivery_env.pickle", "rb") as rf:
    env = pickle.load(rf)
    env.generate_transition_probs()
    env.set_custom_reward_function(gt_rew_vec)
gt_avg_return = get_gt_avg_return(gt_rew_vec=np.array(gt_rew_vec), env=env)

# build random policy
random_pi = build_random_policy(env=env)
V_under_random_pi = iterative_policy_evaluation(
    random_pi, rew_vec=np.array(gt_rew_vec), env=env
)
random_avg_return = np.sum(V_under_random_pi) / env.n_starts

conditions = args.conditions.strip().replace(" ", "").split(",")


min_full_pref_size = float("inf")
for condition in conditions:
    results_sub_dir = ""
    all_segment_pairs = np.load("data/human_data/" + condition + "_full_filtered_segment_pairs.npy",allow_pickle=True)
    min_full_pref_size = min(min_full_pref_size, len(all_segment_pairs))

print ("min_full_pref_size: ", min_full_pref_size)

for seed in [0,1,2,3,4,5,6,7,8,9]:

    print ("on seed:" + str(seed))
    
    plot_i = 0


    for pref_model in ["regret", "pr"]:
        args.preference_assum = pref_model
        if pref_model == "regret":
            args.LR = 0.5
            args.N_ITERS = 5000            
            args.succ_feats = np.load("data/delivery_mdp/succ_feats_no_gt.npy")
            args.succ_q_feats = None
        else:
            args.LR = 2
            args.N_ITERS = 30000
            args.succ_feats = None
            args.succ_q_feats = None

        for condition in conditions:
            
            all_segment_pairs = np.load("data/human_data/" + condition + "_full_filtered_segment_pairs.npy",allow_pickle=True)
            if args.use_synth_prefs:
                all_preferences = np.load("data/human_data/" + condition + "_gt_"+args.mode+"_"+pref_model+"_prefs_Y.npy")
            else:
                all_preferences = np.load("data/human_data/"+condition + "_full_filtered_Y.npy")
            
            
            all_preferences = format_y(all_preferences, keep_as_list=True)

            assert len(all_preferences) == len(all_segment_pairs)
            
            all_rew_vects = []
            all_scaled_returns = []
            all_partitions = [1,2,5,10,20]
            for n_partitions in all_partitions:
                print (condition + " " + pref_model + " pref. model, n partitions:", n_partitions)
                #------------ build the paritions ------------
                combined = list(zip(all_segment_pairs, all_preferences))
                random.Random(seed).shuffle(combined)
                segment_pairs_copy, prefs_copy = zip(*combined)

                partition_size = int(min_full_pref_size/n_partitions)
                partitioned_segment_pairs = []
                # partitioned_segment_phis = []
                partitioned_prefs = []

                for _ in range(n_partitions):
                    partitioned_segment_pairs.append(segment_pairs_copy[:partition_size])
                    # partitioned_segment_phis.append(segment_phis_copy[:partition_size])
                    partitioned_prefs.append(prefs_copy[:partition_size])

                    if _ < n_partitions-1:
                        combined = list(zip(segment_pairs_copy[partition_size:], prefs_copy[partition_size:]))
                        random.Random(seed).shuffle(combined)
                        segment_pairs_copy, prefs_copy = zip(*combined)
                #-------------------------------------------------------------
                
                partition_rew_vects = []
                partition_scaled_returns = []
                for segment_pairs_i, segment_pairs in enumerate(partitioned_segment_pairs):
                    # segment_phis = partitioned_segment_phis[segment_pairs_i]
                    preferences = partitioned_prefs[segment_pairs_i]


                    segment_phis, all_r, all_ses, _ = get_extended_features(
                        args, segment_pairs, env, gt_rew_vec, seg_length=3
                    )
                    
                    if pref_model == "regret":
                        segment_phis = format_regret_feats(segment_phis, all_ses)
                    
                    # augment the dataset by swapping preferences and segment pairs
                    aX, ay = augment_data(segment_phis, preferences, "arr")
                    
                    
                    # learn a reward function from the dataset
                    rew_vect, _, _, training_time = train(
                        aX=aX,
                        ay=ay,
                        args=args,
                        plot_loss=False,
                        env=env,
                        check_point=False,
                    )

                    _, Q = value_iteration(rew_vec=rew_vect, gamma=0.999, env=env)
                    pi = build_pi(Q, env=env)
                    V_under_gt = iterative_policy_evaluation(pi, rew_vec=np.array(gt_rew_vec), env=env)
                    avg_return = np.sum(V_under_gt) / env.n_starts

                    # scale everything: f(z) = (z-x) / (y-x)
                    scaled_return = (avg_return - random_avg_return) / (
                        gt_avg_return - random_avg_return
                    )

                    partition_rew_vects.append(rew_vect)
                    partition_scaled_returns.append(scaled_return)

                    print ("    scaled_return:", scaled_return)


                all_rew_vects.append(partition_rew_vects)
                all_scaled_returns.append(partition_scaled_returns)
            
            if not os.path.exists("reward_learning_res"):
                os.makedirs("reward_learning_res")
            
            extra_info = ""
            if args.use_synth_prefs:
                extra_info = "_" + args.mode + "_synth_prefs"
            extra_info += "_" + str(seed)
            extra_info += "_" + str(all_partitions)
            with open("reward_learning_res/" + condition + "_" + pref_model + extra_info+ "_scaled_returns", 'wb') as f:
                pickle.dump(all_scaled_returns, f)
            with open("reward_learning_res/" + condition + "_" + pref_model + extra_info+ "_rew_vects", 'wb') as f:
                pickle.dump(all_rew_vects, f)