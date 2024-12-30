import pickle
import numpy as np
import matplotlib.pyplot as plt
from learn_advantage.algorithms.rl_algos import build_random_policy, iterative_policy_evaluation,value_iteration,build_pi

preference_assum = "regret"
preference_condition = "REGRET_UI_"

y = np.load("data/delivery_mdp/human_data/"+preference_condition+"DELIVERY_MDP_human_prefs.npy")
regret_X = np.load("data/delivery_mdp/human_data/"+preference_condition+"DELIVERY_MDP_segment_pair_features_regret_form.npy")
gt_rew_vec = np.array([-1,50,-50,1,-1,-2])
with open("data/delivery_mdp/delivery_env.pickle", "rb") as rf:
    env = pickle.load(rf)
    env.generate_transition_probs()
    env.set_custom_reward_function(gt_rew_vec)

env.set_custom_reward_function(gt_rew_vec)
V,_ = value_iteration(rew_vec =gt_rew_vec,env=env)

plot_x = []
plot_y = []
plot_colors = []
n_incorrect = 0
n_total = 0
for pref, x in zip(y, regret_X):

    regrets = []
    value_diff = []
    pr_diff = []
    for i in range(2):
        pr = np.dot(gt_rew_vec, x[i][:6])
        v_s0 = V[int(x[i][6])][int(x[i][7])]
        v_st = V[int(x[i][8])][int(x[i][9])]

        regrets.append(v_s0 -(pr + v_st))
        value_diff.append(v_st-v_s0)
        pr_diff.append(pr)

    # if value_diff[1] - value_diff[0] > 20:
    #     continue
    
    if preference_assum == "regret":
        if regrets[0] < regrets[1]:
            if pref == 0:
                color = "green"
            else:
                color = "red"
        elif regrets[1] < regrets[0]:
            if pref == 1:
                color = "green"
            else:
                color = "red"
        else:
            if pref == 0.5:
                color = "green"
            else:
                color = "red"
    elif preference_assum == "pr":
        if pr_diff[0] > pr_diff[1]:
            if pref == 0:
                color = "green"
            else:
                color = "red"
        elif pr_diff[1] > pr_diff[0]:
            if pref == 1:
                color = "green"
            else:
                color = "red"
        else:
            if pref == 0.5:
                color = "green"
            else:
                color = "red"

    # if color == "red":
    plot_colors.append(color)
    plot_x.append(pr_diff[0]-pr_diff[1])
    plot_y.append(value_diff[0] - value_diff[1])

    if color == "red":
        n_incorrect += 1
    n_total +=1



fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)


#spine placement data centered
ax.spines['left'].set_position(('data', 0.0))
ax.spines['bottom'].set_position(('data', 0.0))
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.set_ylabel("Difference in (v(s_t) - v(s_0))",labelpad=150)
ax.set_xlabel("Difference in partial returns",labelpad=-50)

print ("# of incorrect prefs:",n_incorrect)
print ("total # of prefs:", n_total)
plt.scatter(plot_x, plot_y, c=plot_colors, alpha=0.3,s=60)
plt.show()