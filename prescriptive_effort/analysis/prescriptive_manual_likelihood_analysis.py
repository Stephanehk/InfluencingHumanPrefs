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

def plot_losses(params, losses, color, label):
    params_str = [str(np.round(param,4)) for param in params]

    ax.set_ylim([0.3, 1.0])
    plt.plot(params_str, losses, color=color, label=label)
    plt.plot([params_str[0], params_str[-1]], [0.69, 0.69], color = "black", linestyle = "--")
    # plt.title(condition + " (" + pref_model + " model)")
    plt.ylabel("Mean Cross Entropy Loss")
    plt.xlabel("Scaling Parameter")
    plt.xticks([params_str[0], params_str[25], params_str[-1]])


args = parse_args(ArgsType.LIKELIHOOD_ANALYSIS)
conditions = args.conditions.strip().replace(" ", "").split(",")

plot_i = 0

for pref_model in ["regret", "pr"]:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["font.family"] = "Times New Roman"
    font_size =20
    plt.rcParams.update({"font.size": font_size})
    plt.rcParams["axes.titlesize"] = font_size

    fig = plt.figure(plot_i)
    plot_i += 1
    ax = fig.add_subplot(1, 1, 1)


    for condition in conditions:

        X = np.load("data/human_data/" + condition + "_full_filtered_X.npy")
        Y = np.load("data/human_data/" + condition + "_full_filtered_Y.npy")
        assert len(X) == len(Y)

        if pref_model == "regret":
            X = format_X_adv(X)
        elif pref_model == "pr":
            X =format_X_pr(X)
        elif pref_model == "er":
            X =format_X_expected_return(X)

        Y = torch.tensor(Y, dtype=torch.float).unsqueeze(0)
        a = 0.01
        r = 1.236
        num_params = 25

        if "Control" in condition:
            color = "gray"
            label = "NO_STATS_TRAINING_UI"
        elif "Pr" in condition:
            color = "red"
            label = "PR_TRAINING_UI"
        elif "Regret" in condition:
            color = "blue"
            label = "REGRET_TRAINING_UI"
        
        params = get_params(a,r,num_params)
        losses = get_losses(X,Y,params)
        plot_losses(params, losses, color, label)

        print (condition + " dataset under " + pref_model + " model min loss:" + str(min(losses)))

    plt.tight_layout()
    new_path = "manual_likelihood_analysis/"
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    plt.savefig(new_path + pref_model + ".png", dpi=300)