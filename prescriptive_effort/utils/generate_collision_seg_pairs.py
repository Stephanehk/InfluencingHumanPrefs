import pickle
import torch
import numpy as np
import random
import json

from learn_advantage.utils.pref_dataset_utils import (
    augment_data,
    generate_synthetic_prefs,
    remove_absorbing_transitions,
)
from learn_advantage.utils.segment_feats_utils import get_extended_features
from learn_advantage.utils.argparse_utils import parse_args


args = parse_args()

force_cpu = args.force_cpu = True
keep_ties = args.keep_ties = True
n_prob_samples = args.n_prob_samples
n_prob_iters = args.n_prob_iters


gamma = args.gamma

mode = "sigmoid" if args.mode == "stochastic" else args.mode

preference_model = args.preference_model  # how we generate prefs
preference_assum = args.preference_assum  # how we learn prefs

use_extended_SF = False
learn_oaf = False
extra_details = args.extra_details


# These parameters provide some extra conditions primarily for debuging. They are hardcoded for now.
generalize_SF = args.generalize_SF = False

args.succ_feats = None
args.succ_q_feats = None
args.pis = None
different_start_states = False
print ("USING DIFFERENT START STATES:",different_start_states)
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

args.use_extended_SF = False

gt_rew_vec = np.array([-1,50,-50,1,-1,-2])
with open("data/delivery_mdp/delivery_env.pickle", "rb") as rf:
    env = pickle.load(rf)
    env.generate_transition_probs()
    env.set_custom_reward_function(gt_rew_vec)


with open('data/mturk_delivery_mdp/delivery_task_ss_early_terminating_segments.json') as f:
    all_segment_pairs = json.load(f)["(0.0, 0.0)"]


unfiltered_X, unfiltered_r, unfiltered_ses, _ = get_extended_features(
                args, all_segment_pairs, env, gt_rew_vec, seg_length=args.seg_length
            )

assert len(all_segment_pairs) == len(unfiltered_X)
all_X = []
all_r = []
all_ses = []
segment_pairs = []
for x_pair_i, x_pair in enumerate(unfiltered_X):
    if (x_pair[0][2] == 1 and x_pair[1][2] == 0) or (x_pair[1][2] == 1 and x_pair[0][2] == 0):
        all_X.append(x_pair)
        all_r.append(unfiltered_r[x_pair_i])
        all_ses.append(unfiltered_ses[x_pair_i])
        segment_pairs.append(all_segment_pairs[x_pair_i])

# generate synthetic preferences using the ground truth reward/value function.
pr_X, synth_max_y, _ = generate_synthetic_prefs(
    args,
    pr_X=all_X,
    rewards=all_r,
    sess=all_ses,
    actions=None,
    states=None,
    mode=mode,
    gt_rew_vec=np.array(gt_rew_vec),
    env=env,
)

#reformat prefs to be scalars instead of lists
synth_max_y = [pref[1] for pref in synth_max_y]

assert len(segment_pairs) == len(pr_X)

combined = list(zip(pr_X, synth_max_y, segment_pairs))
random.shuffle(combined)

pr_X, synth_max_y, segment_pairs = zip(*combined)
pr_X = pr_X[:50]
synth_max_y = synth_max_y[:50]
segment_pairs = segment_pairs[:50]


header = "same_start_state_sheep_vs_non_sheep_"

if preference_assum == "regret":
    np.save("data/delivery_mdp/human_data/"+header+"DELIVERY_MDP_segment_pair_features_regret_form.npy", pr_X)
else:
    np.save("data/delivery_mdp/human_data/"+header+"DELIVERY_MDP_segment_pair_features_pr_form.npy", pr_X)

if preference_model == "regret":
    np.save("data/delivery_mdp/human_data/"+header+"DELIVERY_MDP_synth_prefs_"+mode+"_regret.npy", synth_max_y)
else:
    np.save("data/delivery_mdp/human_data/"+header+"DELIVERY_MDP_synth_prefs_"+mode+"_pr.npy", synth_max_y)

np.save("data/delivery_mdp/human_data/"+header+"DELIVERY_MDP_segment_pairs.npy", segment_pairs)


    #  if args.preference_assum == "regret":
    #         X = np.load("data/delivery_mdp/human_data/"+args.preference_condition+"DELIVERY_MDP_segment_pair_features_regret_form.npy")
    #     elif args.preference_assum == "pr":
    #         X = np.load("data/delivery_mdp/human_data/"+args.preference_condition+"DELIVERY_MDP_segment_pair_features_pr_form.npy")
    #     y = np.load("data/delivery_mdp/human_data/"+args.preference_condition+"DELIVERY_MDP_human_prefs.npy")