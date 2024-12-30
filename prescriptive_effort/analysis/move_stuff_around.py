import numpy as np

conditions = ["NO_STATS_TRAINING_UI", "PR_TRAINING_UI", "REGRET_TRAINING_UI", "PR_GUIDING_UI", "REGRET_GUIDING_UI", "NO_STATS_GUIDING_UI", "PR_GT_UI", "REGRET_GT_UI", "NO_STATS_GT_UI"]
for condition in conditions:

    results_sub_dir = ""
    if "GUIDING" in condition:
        results_sub_dir = "follow_up_study/"

    for mode in ["stoch", "det"]:
        for pref_model in ["regret", "pr"]:

            X = np.load("data/delivery_mdp_prescriptive_effort/" +results_sub_dir+  condition + "_full_filtered_X.npy")

            all_segment_pairs = np.load("data/delivery_mdp_prescriptive_effort/" + results_sub_dir+ condition + "_full_filtered_segment_pairs.npy",allow_pickle=True)
            synth_all_preferences = np.load("data/delivery_mdp_prescriptive_effort/" +  results_sub_dir+condition + "_gt_"+mode+"_"+pref_model+"_prefs_Y.npy")
            all_preferences = np.load("data/delivery_mdp_prescriptive_effort/" +  results_sub_dir+condition + "_full_filtered_Y.npy")

            if "TRAINING" in condition:
                if condition == "NO_STATS_TRAINING_UI":
                    mapped_condition = "Trained-Control"
                if condition == "PR_TRAINING_UI":
                    mapped_condition = "Pr-Trained"
                if condition == "REGRET_TRAINING_UI":
                    mapped_condition = "Regret-Trained"
            elif "GUIDING" in condition:
                if condition == "NO_STATS_GUIDING_UI":
                    mapped_condition = "Question-Control"
                if condition == "PR_GUIDING_UI":
                    mapped_condition = "Pr-Question"
                if condition == "REGRET_GUIDING_UI":
                    mapped_condition = "Regret-Question"
            elif "GT" in condition:
                if condition == "NO_STATS_GT_UI":
                    mapped_condition = "Privileged-Control"
                if condition == "PR_GT_UI":
                    mapped_condition = "Pr-Privileged"
                if condition == "REGRET_GT_UI":
                    mapped_condition = "Regret-Privileged"

            np.save("data/human_data/" + mapped_condition + "_full_filtered_X.npy", X)
            np.save("data/human_data/" + mapped_condition + "_full_filtered_segment_pairs.npy", all_segment_pairs)
            np.save("data/human_data/" + mapped_condition + "_gt_"+mode+"_"+pref_model+"_prefs_Y.npy", synth_all_preferences)
            np.save("data/human_data/"+mapped_condition + "_full_filtered_Y.npy",all_preferences)
            