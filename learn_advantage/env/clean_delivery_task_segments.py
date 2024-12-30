import numpy as np
import json
import pickle 
import torch

from learn_advantage.utils.segment_feats_utils import get_extended_features

gt_rew_vec = np.array([-1,50,-50,1,-1,-2])
with open("data/delivery_mdp/delivery_env.pickle", "rb") as rf:
    env = pickle.load(rf)
    env.generate_transition_probs()
    env.set_custom_reward_function(gt_rew_vec)
env.find_n_starts()

board_name = "delivery_task_ss"
with open("data/mturk_delivery_mdp/" + board_name + "_segments_phis.json", 'rb') as filehandle:
    seg_pairs_phis = json.load(filehandle)["(0.0, 0.0)"]
with open("data/mturk_delivery_mdp/" + board_name + "_segments.json", 'rb') as filehandle:
    seg_pairs = json.load(filehandle)["(0.0, 0.0)"]


seg_pair_i = 0
early_terminating_segs = []
early_terminating_seg_phis = []
early_terminating_segs_i = []

for seg_pair, seg_pair_phi in zip(seg_pairs, seg_pairs_phis):

    terminates_early = False
    for seg in seg_pair:
        state=seg[0]
        for action_n, action in enumerate(seg[1:]):
            state, _, done, reward_feature = env.get_next_state(state, env.find_action_index(action))
            if env.is_terminal(state[0], state[1]) and action_n != 2:
                terminates_early=True
    if terminates_early:
        # print ("Terminates early at:",seg_pair_i)
        early_terminating_segs_i.append(seg_pair_i)
        early_terminating_segs.append(seg_pair)
        early_terminating_seg_phis.append(seg_pair_phi)
    seg_pair_i += 1
print (len(early_terminating_segs_i))
np.save("data/mturk_delivery_mdp/" + board_name + "_early_term_segs.npy", early_terminating_segs_i)

#---------------------------- SETUP ARGUMENTS ------------------------
from learn_advantage.utils.argparse_utils import parse_args

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
#here, we fix an error with formatting early terminating segments; we need to remove the action taken after terminating so that action is not drawn when rendering segment pairs
reformatted_early_terminating_segs = []
for seg_pair, seg_pair_phi in zip(early_terminating_segs,early_terminating_seg_phis):
    new_seg_pair = []
    for seg in seg_pair:
        terminates_early = False
        state=seg[0]
        new_seg = [tuple(state)]
        for action_n, action in enumerate(seg[1:]):
            state, _, done, reward_feature = env.get_next_state(state, env.find_action_index(action))
            if not terminates_early:
                new_seg.append(tuple(action))
            if env.is_terminal(state[0], state[1]) and action_n != 2:
                terminates_early=True
        new_seg_pair.append(new_seg)
    reformatted_early_terminating_segs.append(new_seg_pair)


# print (reformatted_early_terminating_segs[0])
# reformatted_early_terminating_seg_phis, _, _, _ = get_extended_features(
#                 args, reformatted_early_terminating_segs, env, gt_rew_vec, seg_length=3
#             )

seg_pairs_phis = {"(0.0, 0.0)": np.array(early_terminating_seg_phis).tolist()}
# seg_pairs = {"(0.0, 0.0)": np.array(reformatted_early_terminating_segs).tolist()}
seg_pairs = {"(0.0, 0.0)": reformatted_early_terminating_segs}

with open("data/mturk_delivery_mdp/" + board_name + "_early_terminating_segments_phis.json", 'w') as filehandle:
    json.dump(seg_pairs_phis, filehandle)
with open("data/mturk_delivery_mdp/" + board_name + "_early_terminating_segments.json", 'w') as filehandle:
    json.dump(seg_pairs, filehandle)