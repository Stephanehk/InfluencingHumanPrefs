import pickle
import numpy as np
import matplotlib.pyplot as plt
from learn_advantage.algorithms.rl_algos import build_random_policy, iterative_policy_evaluation,value_iteration,build_pi


def main():

    #[1,2,5,10,20,50,100]
    regret_ui_x = [2542, 1271, 508.4, 254.2, 127.1, 50.84, 25.42]
    pr_x = [2545, 1272.5, 509, 254.5, 127.25, 50.9, 25.45]
    no_stats_x = [1812, 906, 362.4, 181.2, 90.6, 36.24, 18.12]

    condition2x = {"REGRET_UI_":regret_ui_x, "PARTIAL_RETURN_UI_":pr_x,"NO_STATS_UI_":no_stats_x, "":no_stats_x}
    # global_xs = [10,100,300,1000,3000]
    n_prefs_stand = [int(1812/3), int(1812/10), int(1812/30), int(1812/100)]
    n_prefs_stand = [str(n_prefs) for n_prefs in n_prefs_stand]

    gt_rew_vec = np.array([-1,50,-50,1,-1,-2])
    with open("data/delivery_mdp/delivery_env.pickle", "rb") as rf:
        env = pickle.load(rf)
        env.generate_transition_probs()
        env.set_custom_reward_function(gt_rew_vec)
    

    vec = [-1,50,-50,1,-1,-2]
    V,Qs = value_iteration(rew_vec = np.array(vec),env=env)
    pi = build_pi(Qs,env=env)
    V_under_gt = iterative_policy_evaluation(pi,rew_vec = np.array(vec),env=env)
    gt_avg_return = np.sum(V_under_gt)/92

    random_pi = build_random_policy(env=env)
    V_under_random_pi = iterative_policy_evaluation(random_pi,rew_vec = np.array(vec),env=env)
    random_avg_return = np.sum(V_under_random_pi)/92

    # partition_sizes = ["1","2","5","10","20","50","100"]
    partition_sizes = ["3", "10", "30", "100"]


    fig, ax = plt.subplots(2)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({"font.size": 12})
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    #"REGRET_UI_", "PARTIAL_RETURN_UI_",
    for i, pref_condition in enumerate(["pr_det_synth_prefs_100NO_STATS_UI_", "regret_det_synth_prefs_100NO_STATS_UI_"]):
        for _, model in enumerate(["regret_regret", "pr_pr"]):
            near_opts = []
            for n in partition_sizes:
               
                preference_model = model.split("_")[0]
                preference_assum = model.split("_")[1]
                avg_returns = np.load("data/results/testing_things/" + pref_condition + preference_model + "_"+preference_assum+"_logistc_lin_partitioned" + "_avg_returns_n_split=" + str(n) + ".npy")
              

                n_near_opt = 0
                for avg_return in avg_returns:
                    scaled_ret = (avg_return - random_avg_return)/(gt_avg_return - random_avg_return)
                    if scaled_ret >= 0.9:
                        n_near_opt += 1
                near_opts.append(n_near_opt/int(n))

            # x_ticks = partition_sizes
            # x_ticks = condition2x[pref_condition]
            x_ticks = n_prefs_stand
          
            if model == "regret_regret":
                color = "blue"
                label = "Regret pref. model"
            elif model == "pr_pr":
                color =  "red"
                label = "Partial return pref. model"

            
            # label = pref_condition[:-1]
            
            print (near_opts)
            ax[i].plot(x_ticks, near_opts, color=color, label=label)

        if "regret" in pref_condition:
            title = "regret labeled prefs."
        elif "pr_synth_prefs_100" in pref_condition:
            title = "partial return labeled prefs."


        ax[i].hlines(y=1, xmin=partition_sizes[0], xmax=partition_sizes[-1], colors='grey', linestyles='--', lw=1)
        ax[i].set_title(title)
        ax[i].set_xlabel("# of preferences per partition")
        ax[i].set_ylabel("% near optimal \n across all partitions")
        ax[i].set_yticks([0,0.25, 0.5,0.75, 1])

        #x_ticks = partition_sizes
        # x_ticks = [10,100,300,1000,3000]
        x_ticks = n_prefs_stand
        ax[i].set_xticks(x_ticks)
        ax[i].set_xlim(x_ticks[0], x_ticks[-1])
        vals =  ax[i].get_yticks()
        ax[i].set_yticklabels(['{:,.0%}'.format(x) for x in vals])
        ax[i].invert_xaxis()

        
    fig.set_size_inches(18.5, 10.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("human_data_test.png", dpi=300)


if __name__ == "__main__":
    main()