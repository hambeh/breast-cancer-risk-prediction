# This code is the implementation source code of the second module,  the adaptive iterative search, 
# of our proposed approach on randomly-generated data
# output: best_indices_cvt_auc_recall3, the best indices found from validation data
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np
import pickle
from sklearn.metrics import auc
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import precision_recall_curve

# fixing seed: important to have same random train and test split as the optimizing
np.random.seed(0)

# loading data
#  creating random genotyped data with the same size as the data used in the original manuscript
# Note heterozygous and homozygous minor are encoded as 0, 1, and 2, respectively.
X_kuopio = np.random.randint(3, size=(696, 125041))
Y_kuopio = np.random.randint(2, size=(696, ))

# XGBoost achieved optimal hyperparameters [log_loss, {learning_rate, max_depth, n_estimators}] 
# for 10 iterative process. 
f = open('best_grid_results_stage1_kuopio_0.pckl', 'rb')
best0 = pickle.load(f)
f.close()

f = open('best_grid_results_stage1_kuopio_1.pckl', 'rb')
best1 = pickle.load(f)
f.close()

f = open('best_grid_results_stage1_kuopio_2.pckl', 'rb')
best2 = pickle.load(f)
f.close()

f = open('best_grid_results_stage1_kuopio_7.pckl', 'rb')
best7 = pickle.load(f)
f.close()

# Stage 2 functions
def cal_XGboost(X_train, Y_train, model, x_test, y_test):
    # fitting an XGBoost model and returning feature importance (gain)
    model_XGboost = clone(model)
    eval_set = [(x_test, y_test)]
    model_XGboost.fit(X_train, Y_train, verbose = False, early_stopping_rounds=(model_XGboost.n_estimators)/10, eval_metric="auc", eval_set=eval_set)
    print('XGboost done')
    return model_XGboost.booster().get_score(importance_type='gain')

def all_results_SVM(XX_train,YY_train,XX_validation,YY_validation,indices,model):
    classifier = clone(model)
#    print(model)
    classifier.fit(XX_train[:,indices], YY_train)
    ts_score = classifier.predict_proba(XX_validation[:,indices])
    precision, recall, _ = precision_recall_curve(YY_validation, ts_score[:,1])
    return ts_score[:,1]


def cal_XGboost_feature_importance(X_train,Y_train, indices, model, X_test, Y_test):
    # # initial sort: algorithm 1 step 1
    model_1 = clone(model)
    eval_set = [(X_test[:,indices], Y_test)]
    model_1.fit(X_train[:,indices], Y_train,  verbose = False, early_stopping_rounds=(model_1.n_estimators)/10, eval_metric="auc", eval_set=eval_set)
    # now sort based on gain
    scores_key_values = model_1.booster().get_score(importance_type='gain')
    index_non_zero = list()
    for i in range(len(scores_key_values.keys())): # getting indices of used features in xgboost in [0, len(indices)]
        index_non_zero.append(np.int64(scores_key_values.keys()[i][1:]))# indices of keys
    sorted_values = np.argsort(scores_key_values.values())[::-1] # argsorting based on gain and getting corresponding top indices.
    from_top_temp = indices[np.array(index_non_zero)[sorted_values]]   # in range [0,125041]
    zir_from_top = np.array(list(set(indices)^set(indices[np.array(index_non_zero)[sorted_values]])))
    from_top = np.concatenate((from_top_temp, zir_from_top), axis=0)
    return from_top

# implementation of algorithm 1
def second_cal_XGboost_feature_importance4(XX_train,YY_train, SNPs_indices_sorted_main, M, K, N, model, X_test, Y_test):
    SNPs_indices_sorted= SNPs_indices_sorted_main.copy()
    model1 = clone(model)
    model2 = clone(model)
    temp = []

    for i in K:

        if(M+i <= len(SNPs_indices_sorted)//2):
            eval_set1 = [(X_test[:,SNPs_indices_sorted[:M+i]], Y_test)]
            model1.fit(XX_train[:,SNPs_indices_sorted[:M+i]], YY_train, verbose = False, early_stopping_rounds=(model1.n_estimators)/10, eval_metric="auc",
            eval_set=eval_set1) # from top

            eval_set2 = [(X_test[:,SNPs_indices_sorted[-M-i:]], Y_test)]
            model2.fit(XX_train[:,SNPs_indices_sorted[-M-i:]], YY_train, verbose = False, early_stopping_rounds=(model2.n_estimators)/10, eval_metric="auc",
            eval_set=eval_set2)# from bottom

            scores4_key_values = model1.booster().get_score(importance_type='gain')
            index_non_zero4 = list()
            for uu in range(len(scores4_key_values.keys())): # getting indices of used features in xgboost in [0, len(indices)]
                index_non_zero4.append(np.int64(scores4_key_values.keys()[uu][1:]))# indices of keys
            sorted_values4 = np.argsort(scores4_key_values.values())[::-1] # argsorting based on gain and getting corresponding top indices.
            M_top = []
            M_top_temp = SNPs_indices_sorted[:M+i][np.array(index_non_zero4)[sorted_values4]]
            zir_top = np.array(list(set(SNPs_indices_sorted[:M+i])^set(SNPs_indices_sorted[:M+i][np.array(index_non_zero4)[sorted_values4]])))
            M_top = np.concatenate((M_top_temp, zir_top), axis=0)

            scores5_key_values = model2.booster().get_score(importance_type='gain')
            index_non_zero5 = list()
            for uu in range(len(scores5_key_values.keys())): # getting indices of used features in xgboost in [0, len(indices)]
                index_non_zero5.append(np.int64(scores5_key_values.keys()[uu][1:]))# indices of keys
            sorted_values5 = np.argsort(scores5_key_values.values())[::-1] # argsorting based on gain and getting corresponding top indices.
            M_bottom_temp = SNPs_indices_sorted[-M-i:][np.array(index_non_zero5)[sorted_values5]]
            zir_bottom = np.array(list(set(SNPs_indices_sorted[-M-i:])^set(SNPs_indices_sorted[-M-i:][np.array(index_non_zero5)[sorted_values5]])))
            M_bottom = np.concatenate((M_bottom_temp, zir_bottom), axis=0)

            # replace m top with m bottom  rankked snps
            SNPs_indices_sorted[:M+i] = M_top # resorting based on new M-top
            SNPs_indices_sorted[len(SNPs_indices_sorted)-M-i:len(SNPs_indices_sorted)] = M_bottom # resoritng based on new M-bottom

            SNPs_indices_sorted[M+i-N:M+i] = M_bottom[0:N] # exchange N best M_bottom with N worst M_top 
            SNPs_indices_sorted[len(SNPs_indices_sorted)-M-i:len(SNPs_indices_sorted)-M-i+N] = M_top[M+i-N:M+i] # exchange N worst M_top with N best M_bottom 
            temp = SNPs_indices_sorted

    return temp

def Tune_stage2(xgboost_scores, X_train, Y_train, X_test, Y_test, model): # From test NOT CV

    model_Tune_stage2 = clone(model)
    average_index_non_zero = list()
    for i in range(len(xgboost_scores.keys())): #getting indices of selected features from training set. Indices are in [0,125041]
        average_index_non_zero.append(np.int64(xgboost_scores.keys()[i][1:]))

    MM = [2,4,6,8,10,20,30] # window size (Algorithm 1 step 2)
    K_increament = [1,2,3,4,5] # adaptively increase window size (Algorithm 1 step 5)
    NN = [1] # let's fix N. algorithm 1 step 4 
    global_returned_sorted = list()
    tot_roc = list()
    # let's fix m, n and k
    SNPs_indices_sorted_main = cal_XGboost_feature_importance(X_train, Y_train, np.array(average_index_non_zero), model_Tune_stage2, X_test, Y_test)# initial sorting
    SNPs_indices_sorted_main = np.int64(SNPs_indices_sorted_main)
    for M in MM:
        for KK in K_increament:
            for N in NN:
                # initial N=sort
                K = []
                K = map(lambda x: x*KK , list(range(len(SNPs_indices_sorted_main)//2)))
                returned_sorted4 = second_cal_XGboost_feature_importance4(X_train, Y_train, SNPs_indices_sorted_main, M, K, N, model_Tune_stage2, X_test, Y_test)
                if(len(returned_sorted4)):
                    selected = np.int64(((np.array(list(range(100)))+1)/float(100))*len(returned_sorted4)) # 1% 2%
                    selected = np.unique(selected)
                    for ii in selected:
                        if(ii!=0):
#                            print(ii)
                            ts_score1=all_results_SVM(X_train,Y_train, X_test, Y_test, returned_sorted4[:ii], model_Tune_stage2)
                            # specific M, K, N, ii and CV
                            print('M: ' + str(M)+ ' K: ' + str(KK) + ' N: ' + str(N) + ' ii: ' +  str(ii))
                            global_returned_sorted.append(returned_sorted4[:ii])
                            precision2, recall2, _ = precision_recall_curve(Y_test, ts_score1)
                            tot_roc.append(auc(recall2, precision2))
#                            print(auc(recall2, precision2))
                            

    best_indices_auc_recall = global_returned_sorted[np.argsort(tot_roc)[::-1][0]]
    return best_indices_auc_recall

def build_XGboost(n_estimatorss,max_depthh,learning_ratee,subsamplee):
    model_x = XGBClassifier(nthread=1,seed=0,n_estimators=n_estimatorss,max_depth=max_depthh,learning_rate=learning_ratee,subsample=subsamplee)
    return model_x


# Tuning
NUM_TRIALS = 10
best_indices_cvt_auc_recall = list()
for i in range(NUM_TRIALS):
    print(i)
    x, x_cv, y, y_cv = train_test_split(X_kuopio,Y_kuopio,test_size=0.2,train_size=0.8,stratify=Y_kuopio,random_state=i)
    # optimizing xgboost parameters: never seen on x_cv and y_cv
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
    # Comparison
    temp_vec_log = [-best0[i][0],-best1[i][0],-best2[i][0]] # lower the better
    temp_n_est = [best0[i][1]['n_estimators'],best1[i][1]['n_estimators'],best2[i][1]['n_estimators']]
    temp_max_depth = [best0[i][1]['max_depth'],best1[i][1]['max_depth'],best2[i][1]['max_depth']]
    temp_lr = [best0[i][1]['learning_rate'],best1[i][1]['learning_rate'],best2[i][1]['learning_rate']]
    inx_best = np.argsort(temp_vec_log)[0]

    temp_vec_log1 = [-best7[i][0]] # lower the better
    temp_subsample = [best7[i][1]['subsample']]
    inx_best1 = np.argsort(temp_vec_log1)[0]

    # Fix model for all 5 cvs
    # defining an XGBoost model from obtained optimal hyperparameters
    model = build_XGboost(temp_n_est[inx_best],temp_max_depth[inx_best],temp_lr[inx_best],temp_subsample[inx_best1])
    print(temp_n_est[inx_best],temp_max_depth[inx_best],temp_lr[inx_best],temp_subsample[inx_best1])

    best_indices_cv_auc_recall = list()
    # Important: same train and test split as xgboost optimization codes  by fixing random seed
    for train, test in cv.split(x, y):  
         X_train = x[train]
         Y_train = y[train]
         X_test = x[test]
         Y_test = y[test]
         xgboost_scores1 = cal_XGboost(X_train, Y_train, model, X_test, Y_test)
         best_indices_au_recall = Tune_stage2(xgboost_scores1,X_train,Y_train,X_test,Y_test,model)
         best_indices_cv_auc_recall.append(best_indices_au_recall)

    best_indices_cvt_auc_recall.append(best_indices_cv_auc_recall)

# save the best indices found in each iteration
f = open('best_indices_cvt_auc_recall3.pckl', 'wb')
pickle.dump(best_indices_cvt_auc_recall,f)
f.close()
