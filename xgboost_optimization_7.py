# After finding the optimal n_estimators, max_depth and learning_rate, now subsampling rate is optimized.
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

# fixing seed: important to have same random train and test split as the optimizing
np.random.seed(0)

# loading data
#  creating random genotyped data with the same size as the data used in the original manuscript
# Note heterozygous and homozygous minor are encoded as 0, 1, and 2, respectively.
X_kuopio = np.random.randint(3, size=(696, 125041))
Y_kuopio = np.random.randint(2, size=(696, ))

f = open('best_grid_results_stage1_kuopio_0.pckl', 'rb')
best0 = pickle.load(f)
f.close()

f = open('best_grid_results_stage1_kuopio_1.pckl', 'rb')
best1 = pickle.load(f)
f.close()

f = open('best_grid_results_stage1_kuopio_2.pckl', 'rb')
best2 = pickle.load(f)
f.close()

# Tuning
NUM_TRIALS = 10
subsample = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
param_grid = dict(subsample=subsample)
# Functions
def model_XGboost(n_estimators,max_depth,learning_rate):
    model_x = XGBClassifier(nthread=1,seed=0,n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate)
    return model_x

#model = XGBClassifier(seed=0,nthread=1)
tot_grid_results = list()
best_grid_results = list()
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
    model = model_XGboost(temp_n_est[inx_best],temp_max_depth[inx_best],temp_lr[inx_best])

    grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=5, cv=cv, verbose=1)
    grid_result = grid_search.fit(x, y)
    tot_grid_results.append(grid_result)
    best_grid_results.append([grid_result.best_score_, grid_result.best_params_])
    

# save the hyperparamters    
f = open('tot_grid_results_stage1_kuopio_7.pckl', 'wb')
pickle.dump(tot_grid_results,f)
f.close()

f = open('best_grid_results_stage1_kuopio_7.pckl', 'wb')
pickle.dump(best_grid_results,f)
f.close()

