import numpy as np
from scipy.stats import fisher_exact

# conditions = ["PR_GT_UI", "REGRET_GT_UI", "NO_STATS_GT_UI"]
# conditions =  ["NO_STATS_TRAINING_UI", "PR_TRAINING_UI", "REGRET_TRAINING_UI"]
condition2n_regret_pref_agreements = {}
condition2n_pr_pref_agreements = {}

condition2n_regret_pref_disagreements = {}
condition2n_pr_pref_disagreements = {}

exps = [["Privileged-Control", "Pr-Privileged", "Regret-Privileged"],["Trained-Control", "Pr-Trained", "Regret-Trained"], ["Question-Control", "Pr-Question", "Regret-Question"]]

for conditions in exps:

    for condition in conditions:
    
        
        pr_det_prefs = np.load("data/human_data/" +  condition + "_gt_det_pr_prefs_Y.npy")
        regret_det_prefs = np.load("data/human_data/" +  condition + "_gt_det_regret_prefs_Y.npy")
        human_prefs = np.load("data/human_data/" +  condition + "_full_filtered_Y.npy")

        assert len(human_prefs) == len(regret_det_prefs) == len(pr_det_prefs)


        n_regret_pref_agreements = 0
        n_pr_pref_agreements = 0
        for i, human_pref in enumerate(human_prefs):
            regret_pref = regret_det_prefs[i]
            pr_pref = pr_det_prefs[i]

            if regret_pref == human_pref:
                n_regret_pref_agreements += 1
            if pr_pref == human_pref:
                n_pr_pref_agreements += 1

        if "Control" in condition:
            condition_model = "Control"
        else:
            condition_model = condition.split("-")[0]
        
        condition2n_regret_pref_agreements[condition_model] = n_regret_pref_agreements
        condition2n_pr_pref_agreements[condition_model] = n_pr_pref_agreements

        condition2n_regret_pref_disagreements[condition_model] = len(human_prefs) - n_regret_pref_agreements
        condition2n_pr_pref_disagreements[condition_model] = len(human_prefs) - n_pr_pref_agreements

    regret_cond_table = [[condition2n_regret_pref_agreements["Regret"], condition2n_regret_pref_agreements["Control"]], [condition2n_regret_pref_disagreements["Regret"], condition2n_regret_pref_disagreements["Control"]]]
    pr_cond_table = [[condition2n_pr_pref_agreements["Pr"], condition2n_pr_pref_agreements["Control"]], [condition2n_pr_pref_disagreements["Pr"], condition2n_pr_pref_disagreements["Control"]]]

    regret_res = fisher_exact(regret_cond_table, alternative='two-sided')
    pr_res = fisher_exact(pr_cond_table, alternative='two-sided')

    print ("\n====== " + condition.split("-")[1] + " Expirement======\n")
    print ("Influencing humans towards regret:")
    print ("    Accuracy of noiseless regret model:", condition2n_regret_pref_agreements["Regret"]/(condition2n_regret_pref_agreements["Regret"] + condition2n_regret_pref_disagreements["Regret"]))
    print ("    Fisher exact p-valueL", regret_res.pvalue)
    print ("\n")
    print ("Influencing humans towards partial return:", condition2n_pr_pref_agreements["Pr"]/(condition2n_pr_pref_agreements["Pr"] + condition2n_pr_pref_disagreements["Pr"]))
    print ("    Accuracy of noiseless regret model:")
    print ("    Fisher exact p-value:", pr_res.pvalue)
