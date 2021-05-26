from loading import *
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import ElasticNet, Ridge
import numpy as np
import pandas as pd

from cuml import SVR


# svr_test, svr_losses = cv_test(SVR())

# enet_test_preds, enet_losses = cv_test(ElasticNet(alpha=0.002, l1_ratio=0.99, max_iter=10000, 
#                           normalize=True, selection='random', tol=1e-5))
# ridge_test_preds, ridge_losses = cv_test(Ridge(0.001))

#the outputs from the above 3 lines
ridge_losses = [0.14432562046393557,
 0.1518621332334051,
 0.15199234197025407,
 0.182002015241784,
 0.1773679018606814]

enet_losses = [0.1466418276102861,
 0.15159310188684474,
 0.15164378498766481,
 0.1823087905152352,
 0.17763578983352862]


svr_losses = [0.14452686132265957,
 0.15536270536948454,
 0.15517225770751503,
 0.18654138255008823,
 0.18036466196946171]

importance_mat = pd.DataFrame()

for i in range(5):
    importance_mat[targets[i]] = pd.Series([1/ridge_losses[i], 1/enet_losses[i], 1/svr_losses[i]])
    
for i in range(5):
    sums = importance_mat[targets[i]].sum()
    importance_mat[f'ridge_{targets[i]}'] = (1/ridge_losses[i])/sums
    importance_mat[f'enet_{targets[i]}'] = (1/enet_losses[i])/sums
    importance_mat[f'svr_{targets[i]}'] = (1/svr_losses[i])/sums

importance_mat = importance_mat.drop(targets, axis=1)
importance_mat = importance_mat.drop([0, 1], axis=0)