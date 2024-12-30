import numpy as np
import json
import argparse

from learn_advantage.env.grid_world import GridWorldEnv
from learn_advantage.algorithms.rl_algos import value_iteration
from learn_advantage.utils.segment_feats_utils import get_extended_features
from learn_advantage.utils.pref_dataset_utils import generate_synthetic_prefs

#---------------------------- CREATE EXAMPLE BOARD -----------------------

height = 10
width = 10
board_name = "pref_practice_2"
env = GridWorldEnv(None, height, width)

# board = [[0,0,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0,0], [5,5,5,5,0,0,0,0,0,0], [5,5,5,5,0,0,4,4,4,0], [0,0,0,0,0,0,4,4,4,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]]
board = [[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]]
manual_segment_pairs = [[[(4,5),[0,1],[1,0],[1,0]], [(4,5),[-1,0],[-1,0],[-1,0]]]]

# board = [[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,5,5,5,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,5,5,5,0], [0,0,0,0,0,0,0,0,0,0]]

# board = [[0,0,0,0,0,0,0,0,0,0], [0,0,5,5,5,5,5,5,1,0], [0,0,5,5,5,5,5,5,0,0], [0,0,5,5,5,5,5,5,0,0], [0,0,5,5,5,5,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]]
# manual_segment_pairs = [[[(2,6),[1,0],[1,0],[0,1]], [(2,6),[1,0],[0,1],[0,1]]], [[(3,1),[0,1],[0,1],[0,1]], [(3,1),[1,0],[1,0],[0,1]]], [[(3,2),[0,1],[0,1],[0,1]], [(3,2),[1,0],[1,0],[0,1]]], [[(2,2),[-1,0],[0,1],[0,1]], [(2,2),[-1,0],[-1,0],[0,1]]]]

# board = [[0,0,0,0,0,0,0,0,0,0], [0,0,5,5,5,5,5,5,1,0], [0,0,5,5,5,5,5,5,0,0], [0,0,5,5,5,5,5,5,0,0], [0,0,5,5,5,5,5,0,0,0], [0,0,5,5,5,5,5,0,0,0], [0,0,5,5,5,5,5,0,0,0], [0,0,5,5,5,5,5,0,0,0], [0,0,5,5,5,5,5,0,0,0], [0,0,5,5,5,5,5,0,0,0]]
# manual_segment_pairs = [[(1,1),[0,1],[0,1],[0,1]], [(1,1),[1,0],[1,0],[0,1]]], [[(4,7),[-1,0],[-1,0],[-1,0]], [(4,7),[-1,0],[0,1],[-1,0]]]

# board = [[0,1,0,0,0,0,0,3,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]]
# manual_segment_pairs = [[[(0,4),[0,1],[0,1],[0,1]], [(0,4),[0,-1],[0,-1],[0,-1]]]]

# board = [[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,3,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]]
# manual_segment_pairs = [[[(4,4),[1,0],[1,0],[1,0]], [(4,4),[-1,0],[-1,0],[-1,0]]]]



# board = [[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0],  [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,1,5,5,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]]
# # manual_segment_pairs = [[[(6,5), [0,-1], [0,-1], [0,-1]], [(6,5), [-1,0], [0,-1], [0,-1]]]]
# manual_segment_pairs = [[[(3,1), [1,0], [1,0], [0,1]], [(3,1), [1,0], [1,0], [0,1]]]]



# n_segment_pairs = 25
n_segment_pairs = 30
add_manual_segment_pairs = True

for h in range(height):
    for w in range(width):
        env.board[h][w] = board[h][w]

# rew_vec = np.array([-1,50,-50,1,-1,-2])
rew_vec = np.array([-1,50,-50,1,-1,-2])

reward_fn = env.set_custom_reward_function(rew_vec)
env.find_n_starts()

# print (reward_fn)
# assert False
V, Qs = value_iteration(rew_vec=np.array(rew_vec), gamma=1, env=env)
# print (V.shape)


with open("data/new_mdps/" + board_name + "_board.json", 'w') as filehandle:
    json.dump(env.board.tolist(), filehandle)
with open("data/new_mdps/" + board_name + "_rewards_function.json", 'w') as filehandle:
    json.dump(np.array(env.reward_function).tolist(), filehandle)
with open("data/new_mdps/" + board_name + "_value_function.json", 'w') as filehandle:
    json.dump(V.tolist(), filehandle)

#---------------------------- CREATE EXAMPLE SEGMENT PHIS-----------------------
parser = argparse.ArgumentParser(description="PyTorch RL trainer")
args = parser.parse_args()
args.generalize_SF = False
args.use_extended_SF = False
args.gamma = 0.999

# seg_pairs = {"(0.0, 0.0)": [[[[1,4], [0,-1], [0,-1], [0,-1]], [[1,4], [0,-1], [1,0], [1,0]], 0.0, 0.0, [False, False]], [[[1,4], [0,-1],[0,-1],[0,-1]], [[1,4], [0,1], [0,1], [0,1]], 0.0, 0.0, [False, False]], [[[3,4], [0,-1],[0,-1],[0,-1]], [[3,4], [-1,0], [-1,0], [0,1]], 0.0, 0.0, [False, False]]], "(1.0,0.0)":[[[[5,1],[-1,0],[-1,0],[-1,0]], [[5,1], [-1,0], [0,-1], [0,-1]],0.0, 0.0, [False, False]], [[[6,4], [-1,0], [-1,0], [-1,0]], [[6,4], [1,0], [1,0], [1,0]],0.0, 0.0, [False, False]], [[[4,5], [0,1], [0,1], [1,0]], [[4,5], [-1,0], [-1,0], [-1,0]],0.0, 0.0, [False, False]]]}
# seg_pairs_phis = {}
# for key in seg_pairs:
#     all_X, all_r, all_ses, visited_states = get_extended_features(args, seg_pairs[key], env=env, gt_rew_vec=rew_vec, seg_length=3)
#     seg_pairs_phis[key] = np.array(all_X).tolist()

# with open("data/new_mdps/" + board_name + "_segments_phis.json", 'w') as filehandle:
#     json.dump(seg_pairs_phis, filehandle)
# with open("data/new_mdps/" + board_name + "_segments.json", 'w') as filehandle:
#     json.dump(seg_pairs, filehandle)
# #---------------------------- CREATE QUIZ SEGMENT PHIS-----------------------

segment_pairs = []

#collect double as many segment pairs in case we want to get rid of some
for segment in range(n_segment_pairs):
    a_index = np.random.randint(low=0,high=4)
    print (a_index)
    #sample start state
    x = np.random.randint(low=0, high = env.height)
    y = np.random.randint(low=0, high = env.width)
    while env.is_terminal(x,y) or env.is_blocked(x,y):
        x = np.random.randint(low=0, high = env.height)
        y = np.random.randint(low=0, high = env.width)
    segment_1 = [(x,y)]
    segment_2 = [(x,y)]
    for _ in range(3):
        segment_1.append(tuple(env.actions[np.random.randint(low=0,high=4)]))
        segment_2.append(tuple(env.actions[np.random.randint(low=0,high=4)]))
    segment_pairs.append([segment_1,segment_2])
if add_manual_segment_pairs:
    segment_pairs.extend(manual_segment_pairs)


# #these are only the segment pairs we ended up keeping:
# segment_pairs = [segment_pairs[0], segment_pairs[4], segment_pairs[5], segment_pairs[6], segment_pairs[7],segment_pairs[10]]

all_X, all_r, all_ses, visited_states = get_extended_features(args, segment_pairs, env=env, gt_rew_vec=rew_vec, seg_length=3)

args.keep_ties = True
args.include_actions = False

args.preference_assum = "pr"
args.preference_model = "pr"

_, pr_prefs, _ = generate_synthetic_prefs(
                args,
                pr_X=all_X,
                rewards=all_r,
                sess=all_ses,
                actions=None,
                states=None,
                mode="deterministic",
                gt_rew_vec=np.array(rew_vec),
                env=env,
            )
args.preference_assum = "regret"
args.preference_model = "regret"

regret_X, regret_prefs, _ = generate_synthetic_prefs(
                args,
                pr_X=all_X,
                rewards=all_r,
                sess=all_ses,
                actions=None,
                states=None,
                mode="deterministic",
                gt_rew_vec=np.array(rew_vec),
                env=env,
            )


seg_pairs_phis = {"(0.0, 0.0)": np.array(all_X).tolist()}
seg_pairs = {"(0.0, 0.0)": np.array(segment_pairs).tolist()}
pr_prefs = {"(0.0, 0.0)": np.array(pr_prefs).tolist()}
regret_prefs = {"(0.0, 0.0)": np.array(regret_prefs).tolist()}

with open("data/new_mdps/" + board_name + "_segments_phis.json", 'w') as filehandle:
    json.dump(seg_pairs_phis, filehandle)
with open("data/new_mdps/" + board_name + "_segments.json", 'w') as filehandle:
    json.dump(seg_pairs, filehandle)

with open("data/new_mdps/" + board_name + "_pr_prefs.json", 'w') as filehandle:
    json.dump(pr_prefs, filehandle)
with open("data/new_mdps/" + board_name + "_regret_prefs.json", 'w') as filehandle:
    json.dump(regret_prefs, filehandle)