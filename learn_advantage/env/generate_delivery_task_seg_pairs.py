import numpy as np
import json
import argparse
import pickle
import itertools
import torch

from learn_advantage.env.grid_world import GridWorldEnv
from learn_advantage.algorithms.rl_algos import value_iteration
from learn_advantage.utils.segment_feats_utils import get_extended_features
from learn_advantage.utils.pref_dataset_utils import generate_synthetic_prefs
from learn_advantage.utils.argparse_utils import parse_args


#---------------------------- SETUP ARGUMENTS ------------------------
args = parse_args()

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

#---------------------------- SETUP ARGUMENTS -----------------------

def flatten(arr):
    flattened = []
    for a in arr:
        assert len(a) == 2
        flattened.append(a[0])
        flattened.append(a[1])
    return flattened
height = 10
width = 10
n_pairs_per_state = 50
board_name = "delivery_task_ss"

gt_rew_vec = np.array([-1,50,-50,1,-1,-2])
with open("data/delivery_mdp/delivery_env.pickle", "rb") as rf:
    env = pickle.load(rf)
    env.generate_transition_probs()
    env.set_custom_reward_function(gt_rew_vec)
env.find_n_starts()
V, Qs = value_iteration(rew_vec=np.array(gt_rew_vec), gamma=1, env=env)


task_seg_pairs = []
for x in env.row_iter():
    for y in env.column_iter():
        if env.is_terminal(x,y) or env.is_blocked(x,y):
            continue
        
        n_pairs = 0
        seen_seg_pairs =[]
        while n_pairs < n_pairs_per_state:
            segment_1 = [(x,y)]
            segment_2 = [(x,y)]
            for _ in range(3):
                segment_1.append(tuple(env.actions[np.random.randint(low=0,high=4)]))
                segment_2.append(tuple(env.actions[np.random.randint(low=0,high=4)]))
            # print (segment_1)
            
            flattened_seg_pair = flatten(segment_1)
            flattened_seg_pair.extend(flatten(segment_2))
            flattened_seg_pair = tuple(flattened_seg_pair)
            if flattened_seg_pair not in seen_seg_pairs:
                seen_seg_pairs.append(flattened_seg_pair)
                task_seg_pairs.append([segment_1,segment_2])
                n_pairs += 1
            
print ("# of segment pairs: ", len(task_seg_pairs))

all_X, all_r, all_ses, _ = get_extended_features(
                args, task_seg_pairs, env, gt_rew_vec, seg_length=3
            )


seg_pairs_phis = {"(0.0, 0.0)": np.array(all_X).tolist()}
seg_pairs = {"(0.0, 0.0)": np.array(task_seg_pairs).tolist()}


with open("data/mturk_delivery_mdp/" + board_name + "_segments_phis.json", 'w') as filehandle:
    json.dump(seg_pairs_phis, filehandle)
with open("data/mturk_delivery_mdp/" + board_name + "_segments.json", 'w') as filehandle:
    json.dump(seg_pairs, filehandle)

with open("data/mturk_delivery_mdp/" + board_name + "_board.json", 'w') as filehandle:
    json.dump(env.board, filehandle)
with open("data/mturk_delivery_mdp/" + board_name + "_rewards_function.json", 'w') as filehandle:
    json.dump(np.array(env.reward_function).tolist(), filehandle)
with open("data/mturk_delivery_mdp/" + board_name + "_value_function.json", 'w') as filehandle:
    json.dump(V.tolist(), filehandle)