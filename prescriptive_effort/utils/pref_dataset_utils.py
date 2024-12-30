import numpy as np
import torch

from prescriptive_effort.analysis.models import LogisticRegression

def get_params(a,r,num_params, initial_val=0.0):
    params_neg = [initial_val + (-a * (r ** i)) for i in range(1,num_params+1)]
    params_neg.reverse()
    params_pos = [initial_val + (a * (r ** i)) for i in range(1,num_params+1)]
    params = params_neg
    params.append(initial_val)
    params.extend(params_pos)
    return params

def get_losses(X,Y,params):
    losses = []
    for param in params:
        model = LogisticRegression(input_size=1,bias=False,prob_uniform_resp=False)
        model.linear1.weight = torch.nn.Parameter(torch.tensor([param]).float())
        Y_pred = model(X).unsqueeze(1)
        loss= prefrence_pred_loss(Y_pred, Y).detach().item()
        losses.append(loss)
    return losses

def prefrence_pred_loss(output, target, return_all_losses=False):
    '''
    Calculates the cross entropy loss between a pair of preferences

    Input:
    - output: the predicted preferences
    - target: the ground truth preferences

    Output:
    - the cross entropy loss
    '''
    batch_size = output.size()[0]
    output = torch.squeeze(torch.stack((output, torch.sub(1,output)),axis=2))
    output = torch.clamp(output,min=1e-35,max=None)
    output = torch.log(output)
    target = torch.squeeze(torch.stack((target, torch.sub(1,target)),axis=2))
    res = torch.mul(output,target)
    if return_all_losses:
        return -torch.sum(res, axis=1)
    return -torch.sum(res)/batch_size

    # -sum(log(y*y_hat + (1-y)(1-y_hat)))/n
    #
    

def filter_prefs_by_criteria(X,Y, synth_Y, synth_other_Y, assigned_pref_model_gt_Y, dataset_subset):
    only_disagreeing_X = []
    only_disagreeing_Y = []

    n_correct_agreeing_model = 0

    # dataset_subset = "only_agreeing"
    #full_dataset, only_disagreeing, only_disagreeing_no_indiff, only_agreeing

    for x,human_y, synth_y, other_y, gt_synth_y in zip (X,Y, synth_Y, synth_other_Y, assigned_pref_model_gt_Y):
        #and synth_y != 0.5 and other_y != 0.5:
        # if synth_y != other_y:
        # if synth_y == other_y:
        # if synth_y == other_y:
        if dataset_subset == "full_dataset":
            only_disagreeing_X.append(x)
            only_disagreeing_Y.append(human_y)

            if human_y == gt_synth_y:
                n_correct_agreeing_model +=1
        elif dataset_subset == "only_disagreeing" and synth_y != other_y:
            only_disagreeing_X.append(x)
            only_disagreeing_Y.append(human_y)

            if human_y == gt_synth_y:
                n_correct_agreeing_model +=1
        elif dataset_subset == "only_disagreeing_no_indiff" and synth_y != other_y and synth_y != 0.5 and other_y != 0.5:
            only_disagreeing_X.append(x)
            only_disagreeing_Y.append(human_y)

            if human_y == gt_synth_y:
                n_correct_agreeing_model +=1
        elif dataset_subset == "only_agreeing" and synth_y == other_y:
            only_disagreeing_X.append(x)
            only_disagreeing_Y.append(human_y)

            if human_y == gt_synth_y:
                n_correct_agreeing_model +=1
    return only_disagreeing_X,only_disagreeing_Y, n_correct_agreeing_model

def format_regret_feats(segment_phis, all_ses):
    regret_segment_phis = []
    for pair_i, pair in enumerate(segment_phis):
        regret_pair = []
        for segment_i, segment in enumerate(pair):
            
            segment_with_ses = np.concatenate((segment, all_ses[pair_i][segment_i][0]))
            segment_with_ses = np.concatenate((segment_with_ses, all_ses[pair_i][segment_i][1]))

            regret_pair.append(segment_with_ses)
    
        regret_segment_phis.append(regret_pair)
    return regret_segment_phis