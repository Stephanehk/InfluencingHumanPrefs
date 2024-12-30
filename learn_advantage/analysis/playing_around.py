import pickle
import numpy as np
import gzip


mode = "deterministic"
num_prefs = 1000
model_ = "regret_pr"

for trial in range(100,150):
    
    # random_avg_return = np.load("/Users/stephanehatgiskessell/Documents/legacy_reward_learning_stuff/Regret_Based_Reward_Learning_From_Desktop/Reward_Learning/test_reward_shapped_assum/"+str(trial)+"_random_avg_return.npy")
    # gt_avg_return = np.load("/Users/stephanehatgiskessell/Documents/legacy_reward_learning_stuff/Regret_Based_Reward_Learning_From_Desktop/Reward_Learning/test_reward_shapped_assum/"+str(trial)+"_gt_avg_return.npy")
    oaf_scaled_return = np.load("/Users/stephanehatgiskessell/Documents/legacy_reward_learning_stuff/Regret_Based_Reward_Learning_From_Desktop/Reward_Learning/test_reward_shapped_assum/"+str(trial)+"_"+mode+"_"+str(num_prefs)+"oaf_scaled_return.npy")
    print ("Trial "+str(trial) + " oaf_scaled_return:", oaf_scaled_return)

    