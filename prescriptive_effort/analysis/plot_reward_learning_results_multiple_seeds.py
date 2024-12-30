import numpy as np
import matplotlib.pyplot as plt
import pickle

plot_i = 0
num_partitions = ["1","2","5","10","20"]
num_partitions_times_seeds = ["10","20","50","100","200"]

seeds = [0,1,2,3,4,5,6,7,8,9]

exps = [["Privileged-Control", "Pr-Privileged", "Regret-Privileged"],["Trained-Control", "Pr-Trained", "Regret-Trained"], ["Question-Control", "Pr-Question", "Regret-Question"]]

for conditions in exps:

    for pref_model in ["regret", "pr"]:

        plt.rcParams["font.family"] = "Times New Roman"
        font_size =20
        plt.rcParams.update({"font.size": font_size})
        plt.rcParams["axes.titlesize"] = font_size
        plt.rcParams["axes.spines.right"] = False
        plt.rcParams["axes.spines.top"] = False
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"

        

        fig = plt.figure(plot_i)
        plot_i += 1
        ax = fig.add_subplot(1, 1, 1)

        ax.tick_params(axis='both', which='major', labelsize=font_size)

        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname("Times New Roman") for label in labels]

        x_ticks = num_partitions_times_seeds

        for condition in conditions:
            label = condition.replace("_", "-")
            if "Control" in condition:
                color = "gray"
            elif "Pr" in condition:
                color = "red"
            elif "Regret" in condition:
                color = "blue"

        
            num_near_opts = {}
            for part in num_partitions:
                num_near_opts[part] = 0
            
            num_near_opt = 0
            
            for seed in seeds:

                extra_info = ""
                extra_info += "_" + str(seed)
            
                with open("reward_vecs/" + condition + "_" + pref_model  + extra_info+ "_scaled_returns", 'rb') as f:
                    all_scaled_returns = pickle.load(f)
                for n_partitions_i, n_partitions in enumerate(num_partitions):
                    scaled_rets = all_scaled_returns[n_partitions_i]
            
                    assert len(scaled_rets) == int(n_partitions)
                    for ret in scaled_rets:
                        if ret >= 0.0:
                            num_near_opts[n_partitions] +=1 
            
            for part in num_partitions:
                num_near_opts[part] = num_near_opts[part]/(len(seeds)*int(part))

            plt.plot(x_ticks, list(num_near_opts.values()), label = label, color=color)
        


        ax.hlines(y=1, xmin=x_ticks[0], xmax=x_ticks[-1], colors='grey', linestyles='--', lw=1)
        ax.set_title(pref_model + " pref. model")
        # ax.set_xlabel("Preferences per training set",fontsize=font_size)
        ax.set_xlabel("Number of partitions",fontsize=font_size)

        # ax.set_ylabel("% of partitions in which\nperformance is near optimal",fontsize=font_size)
        ax.set_ylabel("% of partitions in which performance is near optimal over 10 seeds",fontsize=font_size)

        ax.set_yticks([0,0.25, 0.5,0.75, 1])


        ax.set_xticks(x_ticks)
        ax.set_xlim(x_ticks[0], x_ticks[-1])
        ax.set_ylim(-0.05, 1.05)
        vals = ax.get_yticks()
        ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
        ax.invert_xaxis()

        plt.tight_layout()

        fig_width, fig_height = plt.gcf().get_size_inches()
        print(fig_width, fig_height)

        plt.savefig(pref_model + "_"+ str(conditions)+ "_reward_learning_better_rand_res_multiple_seeds.png", dpi=300)
