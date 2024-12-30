from scipy.stats import wilcoxon
import numpy as np

from prescriptive_effort.utils.argparse_utils import parse_args,ArgsType
from prescriptive_effort.analysis.prescriptive_estimator import Estimator


import os
import torch
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from learn_advantage.utils.segment_feats_utils import format_y, format_X_pr, format_X_regret,format_X_adv, format_X_expected_return
from prescriptive_effort.utils.pref_dataset_utils import prefrence_pred_loss, filter_prefs_by_criteria, get_params, get_losses
from prescriptive_effort.analysis.models import LogisticRegression

def get_best_sample_ll(X,Y,a = 0.01,r = 1.236,num_params = 25):
    params = get_params(a,r,num_params)
    losses = get_losses(X,Y,params)
    best_param = params[np.argmin(losses)]
    model = LogisticRegression(input_size=1,bias=False,prob_uniform_resp=False)
    model.linear1.weight = torch.nn.Parameter(torch.tensor([best_param]).float())
    Y_pred = model(X).unsqueeze(1)
    losses= prefrence_pred_loss(Y_pred, Y,return_all_losses=True).detach().numpy()
    return losses
   


exps = [["Pr-Trained", "Regret-Trained"], ["Pr-Question", "Regret-Question"]]

print ("\n==================== Wilcoxon Paired Signed Rank Stats Test ====================\n")

for conditions in exps:

    for condition in conditions:
        if "Pr" in condition:
            pref_model = "pr"
        elif "Regret" in condition:
            pref_model = "regret"
        control_condition = condition.split("-")[1] + "-Control"

        temp_X = np.load("data/human_data/" +  condition + "_full_filtered_X.npy")
        temp_Y = np.load("data/human_data/" +  condition + "_full_filtered_Y.npy")
        segment_pairs = np.load("data/human_data/" +  condition + "_full_filtered_segment_pairs.npy")
        
        temp_control_X = np.load("data/human_data/" +  control_condition + "_full_filtered_X.npy")
        temp_control_Y = np.load("data/human_data/" +  control_condition+ "_full_filtered_Y.npy")
        control_segment_pairs = np.load("data/human_data/" + control_condition + "_full_filtered_segment_pairs.npy")

        X = []
        Y = []
        control_Y = []
        for segment_pair_i, segment_pair in enumerate(segment_pairs):
            is_present = any(np.array_equal(segment_pair, arr) for arr in control_segment_pairs)
            
            if is_present:
                control_index = next((i for i, arr in enumerate(control_segment_pairs) if np.array_equal(segment_pair, arr)), -1)

                X.append(temp_X[segment_pair_i])
                Y.append(temp_Y[segment_pair_i])
                control_Y.append(temp_control_Y[control_index])
        
        if pref_model == "regret":
            X = format_X_adv(X)
        elif pref_model == "pr":
            X =format_X_pr(X)
        elif pref_model == "er":
            X =format_X_expected_return(X)                

        Y = torch.tensor(Y, dtype=torch.float).unsqueeze(0)
        control_Y = torch.tensor(control_Y, dtype=torch.float).unsqueeze(0)

        ll = get_best_sample_ll(X,Y)
        control_ll = get_best_sample_ll(X,control_Y)

        print (condition)
        res = wilcoxon(ll, control_ll)
        print ("W=", res.statistic)
        print ("p=", res.pvalue)
        print ("\n")
        