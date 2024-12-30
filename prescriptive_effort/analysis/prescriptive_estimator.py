import numpy as np
import torch
import random
from sklearn.model_selection import KFold

from learn_advantage.utils.segment_feats_utils import format_y, format_X_pr, format_X_regret,format_X_adv
from prescriptive_effort.utils.pref_dataset_utils import prefrence_pred_loss,get_params, get_losses
from prescriptive_effort.analysis.models import LogisticRegression


class Estimator():
    def __init__(self, pref_assum, min_full_pref_size=1812, n_training_epochs = 1000, lr=0.001) -> None:
        self.pref_assum = pref_assum
        self.min_full_pref_size = min_full_pref_size
        self.n_training_epochs = n_training_epochs
        self.lr= lr
        torch.manual_seed(0)

    def get_partitions(self, n_parts, X_copy, Y_copy):

        assert len (X_copy) == len(Y_copy)

        assert len (X_copy) >= self.min_full_pref_size

        partition_size = int(self.min_full_pref_size/n_parts)
        partitioned_X = []
        partitioned_Y = []
        for _ in range(n_parts):
            partitioned_X.append(X_copy[:partition_size])
            partitioned_Y.append(Y_copy[:partition_size])
            if _ < n_parts-1:
                combined = list(zip(X_copy[partition_size:], Y_copy[partition_size:]))
                random.Random(100).shuffle(combined)
                X_copy, Y_copy = zip(*combined)
        return partitioned_X, partitioned_Y

    def get_ll(self, X_train, y_train):
        model = LogisticRegression(input_size=1,bias=False,prob_uniform_resp=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        for _ in range(self.n_training_epochs):
            model.train()
            model.zero_grad()
            y_pred = model(X_train)
            loss= prefrence_pred_loss(y_pred, y_train)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            y_pred = model(X_train)
            training_d_ll = np.exp(-prefrence_pred_loss(y_pred, y_train).item()*len(y_pred))

            #-sum(log(y*y_hat)+ (1-y)(1-y_hat)))/n
            #sum(log(y*y_hat) + log((1-y)(1-y_hat)))
            #e^

        # print (prefrence_pred_loss(y_pred, y_train))
        # print (len(y_pred))
        # print (training_d_ll)
        # print ("\n")
        return training_d_ll

    def get_ll_man(self, X, Y,a = 0.01,r = 1.236,num_params = 25):
        params = get_params(a,r,num_params)
        losses = get_losses(X,Y,params)

        
        training_d_ll = np.exp(-min(losses)*Y.shape[1])
    
        return training_d_ll
        

    def partitions_bayesian_test(self, X1,Y1, X2,Y2, n_parts):
        all_partition_ratios = []
        for n in n_parts:
            combined = list(zip(X1, Y1, X2, Y2))
            random.Random(100).shuffle(combined)
            X_copy1, Y_copy1, X_copy2, Y_copy2 = zip(*combined)

            partitioned_X1, partitioned_Y1 = self.get_partitions(n, X_copy1, Y_copy1)
            partitioned_X2, partitioned_Y2 = self.get_partitions(n, X_copy2, Y_copy2)

            
            partition_ratios = []
            for partition_i in range(n):
                X_train1 = partitioned_X1[partition_i]
                y_train1 = partitioned_Y1[partition_i]

                X_train2 = partitioned_X2[partition_i]
                y_train2 = partitioned_Y2[partition_i]

                if self.pref_assum == "regret":
                    X_train1 =format_X_adv(X_train1)
                    X_train2 =format_X_adv(X_train2)
                elif self.pref_assum == "pr":
                    X_train1 =format_X_pr(X_train1)
                    X_train2 =format_X_pr(X_train2)

                y_train1= torch.tensor(y_train1, dtype=torch.float).unsqueeze(0)
                y_train2 = torch.tensor(y_train2, dtype=torch.float).unsqueeze(0)

                ll1 = self.get_ll_man(X_train1, y_train1)
                ll2 = self.get_ll_man(X_train2, y_train2)
                
                partition_ratios.append(ll1/ll2)
            all_partition_ratios.append(partition_ratios)
        return all_partition_ratios
        
    def k_folds_eval(self, X,Y, n_folds=10, n_logistic_reg_epochs=10000):
        
        kf = KFold(n_splits=n_folds,random_state = 0,shuffle=True)
        testing_log_losses = []

        for train_index, test_index in kf.split(X):
            self.model = LogisticRegression(input_size=1,bias=False,prob_uniform_resp=False)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
            
            X_train = X[train_index]
            y_train = Y[train_index]
            X_test = X[test_index]
            y_test = Y[test_index]

            if self.pref_assum == "regret":
                X_train =format_X_regret(X_train)
                X_test =format_X_regret(X_test)
            elif self.pref_assum == "pr":
                X_train =format_X_pr(X_train)
                X_test =format_X_pr(X_test)

            y_train= torch.tensor(y_train, dtype=torch.float).unsqueeze(0)
            y_test= torch.tensor(y_test, dtype=torch.float).unsqueeze(0)


            for _ in range(n_logistic_reg_epochs):
                self.model.train()
                self.model.zero_grad()
                y_pred = self.model(X_train)
                loss= prefrence_pred_loss(y_pred, y_train)
                loss.backward()
                self.optimizer.step()
               
            with torch.no_grad():
                y_pred_test = self.model(X_test)

                test_loss = prefrence_pred_loss(y_pred_test, y_test).item()

                print ("train loss:", loss)
                print ("test loss:", test_loss)
                print ("\n")
                testing_log_losses.append(test_loss)

        return np.mean(testing_log_losses)