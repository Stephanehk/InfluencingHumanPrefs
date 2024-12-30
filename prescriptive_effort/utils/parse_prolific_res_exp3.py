import numpy as np
import pandas
import pickle
import os
import matplotlib.pyplot as plt

from learn_advantage.algorithms.rl_algos import value_iteration
from learn_advantage.utils.segment_feats_utils import get_sa_list


class Worker:
    def __init__(self, result_list,gt_rew_vec, V, n_queries, survey_thresh=3.5) -> None:

        self.gt_rew_vec = gt_rew_vec
        self.V = V
        self.result_dict = {}
        for res in result_list:
            res = res.strip().replace("%2C","") #quick cleaning of results
            res = res.split("=")
            self.result_dict[res[0]] = res[1]

        self.prolific_id = self.result_dict["ProlificID"].lower()

        #manually changing problamatic parsing error (not sure how this happened!!)
        if self.prolific_id[0] == "v":
            self.prolific_id = self.prolific_id[1:]

        self.sampleNumber = int(self.result_dict["sampleNumber"])
        self.observationType = int(self.result_dict["observationType"])

        self.n_queries = n_queries
        self.segment_pair_dir = "/Users/stephanehatgiskessell/Documents/Prefrence_Elicitation_Interface/2023_11_29_data_samples/"

        self.fill_in_answ = None
        self.quiz_query_answ_1 = None
        self.quiz_query_answ_2 = None
        self.quiz_query_answ_3 = None
        self.failed_sheep_test_query = None
        self.training_prefs = []
        self.training_segment_phis = []
        self.training_segment_pairs = []

        self.survey_thresh = survey_thresh

        self.process_qas()
        self.survey_score = self.read_survey_results()

    def map_prefs(self, pref):
        if pref == "left":
            return 0
        if pref == "right":
            return 1
        if pref == "same":
            return 0.5
    
    def extract_answ (self, answers, method="first"):
        answers = answers.split(",")
        if method == "first":
            return int(answers[0])
        elif method == "last":
            return int(answers[-1])
        
    def read_survey_results(self):
        
        surver_res = pandas.read_csv('data/post_study_survey_responses/follow_up_study/PROLIFIC Town Van Study Survey (no_stats).csv')

        surver_res = surver_res.map(lambda x: x.strip() if isinstance(x, str) else x)
        
       
        if self.prolific_id == "stephane":
            self.prolific_id = "Stephane"
        
        my_answers = surver_res.loc[surver_res["Prolific ID"] == self.prolific_id]
        if len(my_answers) > 1:
            #means a worker filled out the survey more than once, so let us take the first time
            my_answers = my_answers.head(1)
        # print ("\n")
        score = 0

        print (self.prolific_id)
    
        roadblock_r_answ = my_answers["How much does this item change your SCORE SO FAR? "].item()
        coin_r_answ = my_answers["How much does this item change your SCORE SO FAR? .1"].item()
        goal_r_answ = my_answers["How much does this item change your SCORE SO FAR? .2"].item()
        sheep_r_answ = my_answers["How much does this item change your SCORE SO FAR? .3"].item()
       
        gas_r_answ = my_answers["How much does moving through a blank tile change your SCORE SO FAR? "].item()
        brick_r_answ = my_answers["How much does moving through a brick tile change your SCORE SO FAR? "].item()
        
        #----------------
        a0 = my_answers["What is the goal of this world? (Check all that apply.)"].item().split(";")
        if len(a0) == 1 and a0[0] == "To maximize the score":
            score += 1
        elif len(a0) == 2 and "To maximize the score" in a0 and "To get to a specific location" in a0:
            score += 0.5

        self.passed_house_q = False
        a1 = my_answers["What happens when you try to run into a house?"].item().split(";")
        if len(a1) == 1 and "You incur a gas cost and don't go anywhere." in a1:
            score += 1
            self.passed_house_q = True

        a2 = my_answers["What happens when you run into a sheep? (Check all that apply.)"].item().split(";")
        if len(a2) == 2 and "You are penalized for running into a sheep." in a2 and "The episode ends." in a2:
            score +=1 
        elif len(a2) == 1 and "You are penalized for running into a sheep." in a2:
            score += 0.5
        elif len(a2) == 1 and "The episode ends." in a2:
            score += 0.5

        a3 = my_answers["What happens when you run into a roadblock? (Check all that apply.)"].item().split(";")
        if len(a3) == 1 and "You pay a penalty." in a3:
            score += 1

        a4 = my_answers["Is running into a roadblock ever a good choice in any town?"].item().split(";")
        if len(a4) == 1 and "Yes, in certain circumstances." in a4:
            score += 1

        a5 = my_answers["Is entering the brick area ever a good choice?"].item().split(";") #Note, we removed 1 question from the original post-HIT survey that was used for filtering
        if len(a5) == 1 and "Yes, in certain circumstances" in a5:
            score += 1

        if score >= self.survey_thresh:
            self.passed_survey_test = True
        else:
            self.passed_survey_test = False
        return score

    def passed_filter(self):
        return self.passed_survey_test and self.passed_sheep_test
    
    def process_qas(self):
        self.passed_sheep_test = True
        self.regret_quiz_query_prefs = [[0, 0, 0, 0, 1, 0], [1, 1, 1, 0, 1, 0], [0.5, 0.5, 1, 1, 1, 0]]
        self.pr_quiz_query_prefs = [[0.5, 0.5, 0.5, 0.5, 0.5, 1], [0.5, 0.5, 1, 0.5, 1, 0], [0, 0.5, 1, 0, 1,  1]]

        if self.observationType == 0:
            self.correct_fill_in_answ = [47, 48, 47, 44, 45, 44]
            self.quiz_query_correct_answ_1 = self.regret_quiz_query_prefs[0]
            self.quiz_query_correct_answ_2 = self.regret_quiz_query_prefs[1]
            self.quiz_query_correct_answ_3 = self.regret_quiz_query_prefs[2]
        elif self.observationType == 1:
            self.correct_fill_in_answ = [47, 48, 47]
            self.quiz_query_correct_answ_1 = self.pr_quiz_query_prefs[0]
            self.quiz_query_correct_answ_2 = self.pr_quiz_query_prefs[1]
            self.quiz_query_correct_answ_3 = self.pr_quiz_query_prefs[2]

        
        segment_pairs = np.load(self.segment_pair_dir + "none_sample" + str(self.sampleNumber) + "/segment_pairs.npy")
        segment_phis = np.load(self.segment_pair_dir + "none_sample" + str(self.sampleNumber) + "/segment_phis.npy")

        for key,value in self.result_dict.items():
            if key == "sampleNumber" or key == "observationType" or key == "ProlificID":
                continue
            
            if "fillIn" in key:
                if self.fill_in_answ is None:
                    self.fill_in_answ = []
                self.fill_in_answ.append(self.extract_answ(value))
            elif "1quizQuery" in key:
                if self.quiz_query_answ_1 is None:
                    self.quiz_query_answ_1 = []
                self.quiz_query_answ_1.append(self.map_prefs(value))
            elif "2quizQuery" in key:
                if self.quiz_query_answ_2 is None:
                    self.quiz_query_answ_2 = []
                self.quiz_query_answ_2.append(self.map_prefs(value))
            elif "3quizQuery" in key:
                if self.quiz_query_answ_3 is None:
                    self.quiz_query_answ_3 = []
                self.quiz_query_answ_3.append(self.map_prefs(value))
            else:
                query_num = int(key.replace("query", ""))
                if value != "dis":
                    mapped_pref = self.map_prefs(value)
                    self.training_prefs.append(mapped_pref)
                    self.training_segment_pairs.append(segment_pairs[query_num])
                    self.training_segment_phis.append(segment_phis[query_num])

                    phi1 = segment_phis[query_num][0]
                    phi2 = segment_phis[query_num][1]
                    if phi1[2] == 0 and phi2[2] == 1 and mapped_pref != 0:
                        self.passed_sheep_test = False
                        self.failed_sheep_test_query = query_num
                    if phi1[2] == 1 and phi2[2] == 0 and mapped_pref != 1:
                        self.passed_sheep_test = False
                        self.failed_sheep_test_query = query_num

                if query_num == self.n_queries - 1:
                    #The last segment pair is where we show subjects the van running into a sheep (left) vs. not doing so
                    if self.map_prefs(value) == 1:
                        # self.passed_sheep_test = True
                        pass
                    else:
                        self.passed_sheep_test = False
    
    def check_pref_set(self, pref_answ_1, pref_answ_2, pref_answ_3):

        n_correct_1 = 0
        n_correct_2 = 0
        n_correct_3 = 0

        for i, answ in enumerate(pref_answ_1):
            if answ == self.quiz_query_correct_answ_1[i]:
                n_correct_1 +=1
        for i, answ in enumerate(pref_answ_2):
            if answ == self.quiz_query_correct_answ_2[i]:
                n_correct_2 +=1
        for i, answ in enumerate(pref_answ_3):
            if answ == self.quiz_query_correct_answ_3[i]:
                n_correct_3 +=1
        
        return n_correct_1, n_correct_2, n_correct_3


    def eval_pref_model_understanding(self):

        n_correct_1, n_correct_2, n_correct_3 = self.check_pref_set(self.quiz_query_answ_1, self.quiz_query_answ_2, self.quiz_query_answ_3)
        
        fill_in_error = np.abs(np.subtract(self.fill_in_answ, self.correct_fill_in_answ))
        
    
        return n_correct_1/len(self.quiz_query_correct_answ_1), n_correct_2/len(self.quiz_query_correct_answ_2), n_correct_3/len(self.quiz_query_correct_answ_3), fill_in_error

    def process_pref_data(self):
        r_diffs = []
        vs0_diffs = []
        vs3_diffs = []
        for query_n, segment_pair in enumerate(self.training_segment_pairs):
            # if query_n >= 25:
            #     break

            pr1 = np.dot(self.training_segment_phis[query_n][0],self.gt_rew_vec)
            pr2 = np.dot(self.training_segment_phis[query_n][1],self.gt_rew_vec)
            r_diffs.append(pr2-pr1)

            vs0_1 = self.V[segment_pair[0][0][0]][segment_pair[0][0][1]]
            vs0_2 = self.V[segment_pair[1][0][0]][segment_pair[1][0][1]]
            vs0_diffs.append(vs0_2-vs0_1)

            last_state_1 = get_sa_list(segment_pair[0], env)[1]
            last_state_2 = get_sa_list(segment_pair[1], env)[1]
            vs3_1 = self.V[last_state_1[0]][last_state_1[1]]
            vs3_2 = self.V[last_state_2[0]][last_state_2[1]]
            vs3_diffs.append(vs3_2-vs3_1)

            # print (segment_pair)
            # print (pr1, vs0_1, vs3_1, last_state_1)
            # print (pr2, vs0_2, vs3_2, last_state_2)
            # print ("\n")
        return r_diffs, vs0_diffs, vs3_diffs


with open('data/delivery_mdp_prescriptive_effort/prolific_collected_data_follow_study.txt', 'r') as file:
    all_results = file.read()
all_results = all_results.split("assignmentId=&")
# with open('data/delivery_mdp_prescriptive_effort/stephane_test_data.txt', 'r') as file:
#     all_results = file.read()
# all_results = all_results.split("assignmentId=&")

gt_rew_vec=[-1,50,-50,1,-1,-2]
with open("data/delivery_mdp/delivery_env.pickle", "rb") as rf:
    env = pickle.load(rf)
    env.generate_transition_probs()
    env.set_custom_reward_function(gt_rew_vec)
V,Qs = value_iteration(rew_vec = gt_rew_vec, env=env, gamma=1.0)



r_diffs = {2:[], 3:[], 4:[]}
vs0_diffs = {2:[],3:[], 4:[]}
vs3_diffs = {2:[], 3:[], 4:[]}
segment_pairs = {2:[],3:[], 4:[]}
segment_phis = {2:[], 3:[], 4:[]}
prefs = {2:[], 3:[], 4:[]}
worker_observation_types =  {2:[], 3:[], 4:[]} #for testing

headers = {2:"NO_STATS_GUIDING_UI", 3:"PR_GUIDING_UI", 4:"REGRET_GUIDING_UI"}
n_workers_passed = 0
n_workers_passed_per_cond = {2:0, 3:0, 4:0}
n_workers_total_per_cond  = {2:0, 3:0, 4:0}
n_workers_failed_sheep_house_q  = {2:0, 3:0, 4:0}

n_correct_quiz_answ_dict = {2:[0,0,0], 3:[0,0,0], 4:[0,0,0]}
n_total_workers = {2:0, 3:0, 4:0}
fill_in_error_per_cond =  {2:[], 3:[], 4:[]}
failed_pairs = []
passed_pairs = []
for result in all_results:
    result_list = result.split("&")
    if len (result_list) == 1:
        continue
    
    worker = Worker(result_list,gt_rew_vec, V, n_queries=50)
   
    print (worker.prolific_id)
    print (worker.sampleNumber, worker.observationType)
    print ("Passed filtering:", worker.passed_filter())
    print ("    Survey Score: ", worker.survey_score)
    print ("    Passes Sheep Test: ", worker.passed_sheep_test)
    print ("    Failed sheep test query number: ", worker.failed_sheep_test_query)
    print ("\n")

    if not worker.passed_house_q and not worker.passed_sheep_test:
        n_workers_failed_sheep_house_q[worker.observationType] += 1

    if worker.sampleNumber == 3:
        print ("skipping saving data from worker for sample number 3")
        continue

    n_workers_total_per_cond[worker.observationType] +=1
    if worker.passed_filter():
        n_workers_passed+=1
        n_workers_passed_per_cond[worker.observationType] +=1
        
        worker_pref_data = worker.process_pref_data()
        r_diffs[worker.observationType].extend(worker_pref_data[0])
        vs0_diffs[worker.observationType].extend(worker_pref_data[1])
        vs3_diffs[worker.observationType].extend(worker_pref_data[2])
        prefs[worker.observationType].extend(worker.training_prefs)

        segment_pairs[worker.observationType].extend(worker.training_segment_pairs)
        segment_phis[worker.observationType].extend(worker.training_segment_phis)
        

        worker_X = np.stack([worker_pref_data[0], worker_pref_data[1], worker_pref_data[2]], axis = 1)

        
        

        worker_sub_dir = "data/delivery_mdp_prescriptive_effort/follow_up_study/per_worker_res_" + headers[worker.observationType] + "/"
        if not os.path.exists(worker_sub_dir):
            os.makedirs(worker_sub_dir)

        np.save(worker_sub_dir + "worker_" + str(n_workers_passed_per_cond[worker.observationType]) + "_X", worker_X)
        np.save(worker_sub_dir + "worker_" + str(n_workers_passed_per_cond[worker.observationType]) + "_Y", worker.training_prefs)
        passed_pairs.append((worker.sampleNumber, worker.observationType))
    else:
        failed_pairs.append((worker.sampleNumber, worker.observationType))



print ("# of workers that passed comprehension filter:", n_workers_passed)
print ("# of workers that failed comprehension filter:", len(failed_pairs))
print (n_workers_passed_per_cond)
print (n_workers_total_per_cond)
print ("# of workers who failed the sheep test and the question about the house:")
print (n_workers_failed_sheep_house_q)

for observationType in [2,3,4]:
    X = np.stack([r_diffs[observationType], vs0_diffs[observationType], vs3_diffs[observationType]], axis = 1)
    Y = prefs[observationType]
    
    # print ("observationType: ", observationType)
    # print (worker_observation_types[observationType])
    # print ("=========")

    np.save("data/delivery_mdp_prescriptive_effort/follow_up_study/" + headers[observationType] + "_full_filtered_X", X)
    np.save("data/delivery_mdp_prescriptive_effort/follow_up_study/" + headers[observationType] + "_full_filtered_Y", Y)
    
    np.save("data/delivery_mdp_prescriptive_effort/follow_up_study/" + headers[observationType] + "_full_filtered_segment_pairs", segment_pairs[observationType])
    np.save("data/delivery_mdp_prescriptive_effort/follow_up_study/" + headers[observationType] + "_full_filtered_segment_phis", segment_phis[observationType])

    # np.save("data/delivery_mdp_prescriptive_effort/" + headers[observationType] + "_stephane_test_data_X", X)
    # np.save("data/delivery_mdp_prescriptive_effort/" + headers[observationType] + "_stephane_test_data_Y", Y)
