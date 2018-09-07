#First we oprimize n_estimators = [150, 200], max_depth and learning_rate.
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

# Tuning
NUM_TRIALS = 10
n_estimators = [150, 200]
max_depth = [2, 4, 6, 8]
learning_rate = [0.001, 0.01, 0.1]

param_grid = dict(max_depth=max_depth,n_estimators=n_estimators,learning_rate=learning_rate)
model = XGBClassifier(seed=0,nthread=1)
tot_grid_results = list()
best_grid_results = list()
for i in range(NUM_TRIALS):
    print(i)
    x, x_cv, y, y_cv = train_test_split(X_kuopio,Y_kuopio,test_size=0.2,train_size=0.8,stratify=Y_kuopio,random_state=i)
    # optimizing xgboost parameters: never seen on x_cv and y_cv
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
    grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=5, cv=cv, verbose=1)
    grid_result = grid_search.fit(x, y)
    tot_grid_results.append(grid_result)
    best_grid_results.append([grid_result.best_score_, grid_result.best_params_])
    
# save the hyperparamters    
f = open('tot_grid_results_stage1_kuopio_1.pckl', 'wb')
pickle.dump(tot_grid_results,f)
f.close()

f = open('best_grid_results_stage1_kuopio_1.pckl', 'wb')
pickle.dump(best_grid_results,f)
f.close()