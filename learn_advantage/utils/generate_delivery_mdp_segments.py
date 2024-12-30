import pickle
import torch
import numpy as np
import random

from learn_advantage.utils.pref_dataset_utils import (
    augment_data,
    generate_synthetic_prefs,
    remove_absorbing_transitions,
)
from learn_advantage.utils.segment_feats_utils import get_extended_features
from learn_advantage.utils.argparse_utils import parse_args


args = parse_args()

force_cpu = args.force_cpu
keep_ties = args.keep_ties
n_prob_samples = args.n_prob_samples
n_prob_iters = args.n_prob_iters


gamma = args.gamma

mode = "sigmoid" if args.mode == "stochastic" else args.mode

preference_model = args.preference_model  # how we generate prefs
preference_assum = args.preference_assum  # how we learn prefs

use_extended_SF = args.use_extended_SF
learn_oaf = args.learn_oaf
extra_details = args.extra_details
seg_length = args.seg_length

start_MDP = args.start_MDP
end_MDP = args.end_MDP
MDP_dir = args.MDP_dir
all_num_prefs = [int(item) for item in args.num_prefs.split(",")]
dont_include_absorbing_transitions = args.dont_include_absorbing_transitions
dir_name = args.output_dir_prefix

# These parameters provide some extra conditions primarily for debuging. They are hardcoded for now.
use_val_set = False
generalize_SF = args.generalize_SF = False

args.succ_feats = None
args.succ_q_feats = None
args.pis = None
different_start_states = True
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

np.random.seed(0)
n_segment_pairs = 2000
seg_length =3
segment_pairs = []
start_states = []
for _ in range(n_segment_pairs):
    a_index = np.random.randint(low=0,high=4)
    print (a_index)
    #sample start state
    x = np.random.randint(low=0, high = env.height)
    y = np.random.randint(low=0, high = env.width)
    while env.is_terminal(x,y) or env.is_blocked(x,y):
        x = np.random.randint(low=0, high = env.height)
        y = np.random.randint(low=0, high = env.width)
    start_states.append((x,y))

    segment_1 = [(x,y)]
    segment_2 = [(x,y)]
    for _ in range(3):
       
        segment_1.append(tuple(env.actions[np.random.randint(low=0,high=4)]))
        segment_2.append(tuple(env.actions[np.random.randint(low=0,high=4)]))
    segment_pairs.append([segment_1,segment_2])


if different_start_states:
    # segment_pairs = np.array(segment_pairs)
    # print (segment_pairs.shape)

    # segment_pairs_left = segment_pairs[:,:1,:,]
    # segment_pairs_right = segment_pairs[:,1:,:,]

   
    # random.Random(100).shuffle(segment_pairs_left)
    # random.Random(100).shuffle(segment_pairs_right)

    # segment_pairs = np.concatenate((segment_pairs_left, segment_pairs_right), axis=1)

    # for pair in segment_pairs:
    #     print (pair)
    shuffled_segment_pairs = []
    for pair in segment_pairs:
        i_of_interest = np.random.choice([0,1])
        segment = pair[i_of_interest]
        new_start_state_i =  np.random.choice(list(range(len(start_states))))
        while start_states[new_start_state_i] == segment[0]:
            new_start_state_i =  np.random.choice(list(range(len(start_states))))
        new_start_state = start_states[new_start_state_i]
        start_states.pop(new_start_state_i)
        segment[0] = new_start_state
        shuffled_segment_pairs.append([segment, pair[1-i_of_interest]])
    segment_pairs = shuffled_segment_pairs
       

print (segment_pairs[0])
all_X, all_r, all_ses, _ = get_extended_features(
                args, segment_pairs, env, gt_rew_vec, seg_length=seg_length
            )

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

if different_start_states:
    header = "different_start_state_"
else:
    header = "same_start_state_"

if preference_assum == "regret":
    np.save("data/delivery_mdp/human_data/"+header+"DELIVERY_MDP_segment_pair_features_regret_form.npy", pr_X)
else:
    np.save("data/delivery_mdp/human_data/"+header+"DELIVERY_MDP_segment_pair_features_pr_form.npy", pr_X)

if preference_model == "regret":
    np.save("data/delivery_mdp/human_data/"+header+"DELIVERY_MDP_human_prefs_"+mode+"_regret.npy", synth_max_y)
else:
    np.save("data/delivery_mdp/human_data/"+header+"DELIVERY_MDP_human_prefs_"+mode+"_pr.npy", synth_max_y)


    #  if args.preference_assum == "regret":
    #         X = np.load("data/delivery_mdp/human_data/"+args.preference_condition+"DELIVERY_MDP_segment_pair_features_regret_form.npy")
    #     elif args.preference_assum == "pr":
    #         X = np.load("data/delivery_mdp/human_data/"+args.preference_condition+"DELIVERY_MDP_segment_pair_features_pr_form.npy")
    #     y = np.load("data/delivery_mdp/human_data/"+args.preference_condition+"DELIVERY_MDP_human_prefs.npy")