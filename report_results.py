
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

# fixing seed: important to have same random train and test split as the optimizing
np.random.seed(0)

# loading data
# Creating random genotyped data with the same size as the data used in the original manuscript
# Note heterozygous and homozygous minor are encoded as 0, 1, and 2, respectively.
X_kuopio = np.random.randint(3, size=(696, 125041))
Y_kuopio = np.random.randint(2, size=(696, ))

# loading best indices from the output of the clean_second_module.py
f = open('best_indices_cvt_auc_recall3.pckl', 'rb')
best_indices = pickle.load(f)
f.close()

indices_new = []
temp = list()
for j in range(len(best_indices)):
    for k in range(len(best_indices[j])):
#        temp.append(len(best_indices[j][k]))
        indices_new.append(list(best_indices[j][k]))
        
indices_new1 = np.unique(np.concatenate(indices_new))
print(len(indices_new1))

def all_results_SVM(XX_train,YY_train,XX_validation,YY_validation,indices):

    classifier = svm.SVC(probability=True, random_state=3, kernel='linear', C=1.5, class_weight='balanced')
    classifier.fit(XX_train[:,indices], YY_train)
    ts_score = classifier.predict_proba(XX_validation[:,indices])
    return ts_score[:,1]

NUM_TRIALS = 10
counter = -1

tot_average_precisionTR = list()
tot_average_precisionDev = list()
tot_average_precisionTS = list()

indices_ID = range(X_kuopio.shape[0])
for i in range(NUM_TRIALS):
    print(i)
    x, x_cv, y, y_cv, indices_x,indices_x_cv = train_test_split(X_kuopio,Y_kuopio,indices_ID,test_size=0.2,train_size=0.8,stratify=Y_kuopio,random_state=i)
    # optimizing xgboost parameters: never seen on x_cv and y_cv
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
    
    log_cv = list()
    eer_cv = list()
    sens_cv = list()
    spec_cv = list()
    overall_acc_cv = list()
    farr_cv = list()
    tprr_cv = list()
    auc_score_cv = list()  
    prec_cv = list()
    gmean_cv = list()
    fscore_cv = list()

    for train, test in cv.split(x, y):  # accessing train and test for stage 2 parameter tuning 
         X_train = x[train]
         Y_train = y[train]
         X_test = x[test]
         Y_test = y[test]
         counter = counter+1
         if(len(indices_new[counter])):

             # training
             ts_scoreL1=all_results_SVM(X_train, Y_train, X_train, Y_train, indices_new[counter])           
             tot_average_precisionTR.append(average_precision_score(Y_train, ts_scoreL1))             
             # dev
             ts_scoreL1=all_results_SVM(X_train, Y_train, X_test, Y_test, indices_new[counter])           
             tot_average_precisionDev.append(average_precision_score(Y_test, ts_scoreL1))               
             # test             
             ts_scoreL1=all_results_SVM(X_train, Y_train, x_cv, y_cv, indices_new[counter])           
             tot_average_precisionTS.append(average_precision_score(y_cv, ts_scoreL1))
             

print(str('Train Average precision: ')  + str(np.mean(tot_average_precisionTR)*100)+ str('std: ') + str(np.std(tot_average_precisionTR)))
print(str('Dev Average precision: ')  + str(np.mean(tot_average_precisionDev)*100)+ str('std: ') + str(np.std(tot_average_precisionDev)))
print(str('Test Average precision: ')  + str(np.mean(tot_average_precisionTS)*100)+ str('std: ') + str(np.std(tot_average_precisionTS)))



