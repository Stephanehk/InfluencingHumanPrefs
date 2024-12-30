from scipy.stats import mannwhitneyu
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


seed = 0
control_condition = "Privileged-Control"
print ("\n==================== Mann-Whitney U Stats Test ====================\n")
for condition in ["Pr-Privileged", "Regret-Privileged"]:
    if "Pr" in condition:
        pref_model = "pr"
    elif "Regret" in condition:
        pref_model = "regret"
    

    X = np.load("data/human_data/"+  condition + "_full_filtered_X.npy")
    Y = np.load("data/human_data/" +  condition + "_full_filtered_Y.npy")
    segment_pairs = np.load("data/human_data/" +  condition + "_full_filtered_segment_pairs.npy")
    
    control_X = np.load("data/human_data/" +  control_condition + "_full_filtered_X.npy")
    control_Y = np.load("data/human_data/" +  control_condition+ "_full_filtered_Y.npy")
    control_segment_pairs = np.load("data/human_data/" + control_condition + "_full_filtered_segment_pairs.npy")

    if pref_model == "regret":
        X = format_X_adv(X)
        control_X = format_X_adv(control_X)
    elif pref_model == "pr":
        X =format_X_pr(X)
        control_X =format_X_pr(control_X)
    elif pref_model == "er":
        X =format_X_expected_return(X)    
        control_X =format_X_expected_return(control_X)                

    Y = torch.tensor(Y, dtype=torch.float).unsqueeze(0)
    control_Y = torch.tensor(control_Y, dtype=torch.float).unsqueeze(0)

    ll = get_best_sample_ll(X,Y)
    control_ll = get_best_sample_ll(control_X,control_Y)

    print (condition)
    res = mannwhitneyu(ll, control_ll)
    print ("U=", res.statistic)
    print ("p=", res.pvalue)
    print ("\n")
    