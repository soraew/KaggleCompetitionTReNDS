from loading import *
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import ElasticNet, Ridge
import numpy as np
import pandas as pd

from weights import importance_mat

# stacking train and pred for test set

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import cudf as cd    #rapids
from cuml import SVR #rapids

#first, split for validation of meta model
# meta_val, train = train_test_split(train_df)


features = fnc_features + loading_features # excluded IC_20
train = train_df   #not validating meta model for now
test = test_df[features]


folds = 5
kf = KFold(n_splits = folds, shuffle=True, random_state=0)#kfold cross val on train


sub = pd.DataFrame()   # for storing meta preds on test set


ridge_test = pd.DataFrame()
enet_test = pd.DataFrame()
svr_test = pd.DataFrame()

meta_train_df = pd.DataFrame()

for target in targets:
    
    print('\n target : ', target)
    
    
    ridge_test_preds = pd.DataFrame()
    enet_test_preds = pd.DataFrame()
    svr_test_preds = pd.DataFrame()
    
    meta_test_preds = pd.DataFrame()# for submission
    meta_test_train = pd.DataFrame()# dataset used to train meta model on test set
    
    train = train.loc[train[target].notnull()] # using traindf without null target values
    for fold, (train_ind, val_ind) in enumerate(kf.split(train)):
        print('fold : ', fold)
        
        base_train_X, base_val_X = train.iloc[train_ind][features], train.iloc[val_ind][features]
        base_train_y, base_val_y = train.iloc[train_ind][target], train.iloc[val_ind][target]
        
       
        
        # base models
        ridge = Ridge(alpha=0.001)
        enet = ElasticNet(alpha=0.002, l1_ratio=0.99, max_iter=10000, 
                          normalize=True, selection='random', tol=1e-5)
        if target=='age':
            c = 100
        else:
            c = 10
        svr = SVR(C=c, cache_size=3000.0)
        
        
        
        #fit for base models
        ridge.fit(base_train_X, base_train_y)
        enet.fit(base_train_X, base_train_y)
        svr.fit(cd.DataFrame(base_train_X), cd.Series(base_train_y))
        
        #predict on val for base models
        ridge_pred = ridge.predict(base_val_X)
        enet_pred = enet.predict(base_val_X)
        svr_pred = svr.predict(cd.DataFrame(base_val_X))
        svr_pred = np.asarray(svr_pred)
        svr_pred = pd.Series(svr_pred)
        
        #predict on test for base models
        ridge_test_pred = ridge.predict(test)
        enet_test_pred = enet.predict(test)
        svr_test_pred = svr.predict(cd.DataFrame(test))
        svr_test_pred = np.asarray(svr_test_pred)
        svr_test_pred = pd.Series(svr_test_pred)
        
        
        
        
        #fit for meta model
        r_w = importance_mat[f'ridge_{target}'].values  #weights 
        e_w = importance_mat[f'enet_{target}'].values
        s_w = importance_mat[f'svr_{target}'].values
        meta_train_X = ridge_pred*r_w + enet_pred*e_w + svr_pred*s_w
        
        meta_train_X = np.array(meta_train_X).reshape(-1, 1)
        param = {'num_leaves':80, 'metric':'auc', 'objective':'regression'}
        label = base_val_y
        train_data = lgb.Dataset(meta_train_X, label=label)
        meta_model = lgb.train(param, train_data)
        
        #predict for test set with meta model
        meta_test_o = ridge_test_pred*r_w + enet_test_pred*e_w + svr_test_pred*s_w
        meta_test = np.array(meta_test_o).reshape(-1, 1)
        print('meta_shape', meta_test.shape)
        meta_test_pred = meta_model.predict(meta_test)
        
        
        
        
        #concat for taking mean later
        ridge_test_preds = pd.concat([ridge_test_preds, pd.Series(ridge_test_pred)], axis=1)
        enet_test_preds = pd.concat([enet_test_preds, pd.Series(enet_test_pred)], axis=1)
        svr_test_preds = pd.concat([svr_test_preds, pd.Series(svr_test_pred)], axis=1)
        meta_test_preds = pd.concat([meta_test_preds, pd.Series(meta_test_pred)], axis=1)
        
        meta_test_train = pd.concat([meta_test_train, pd.Series(np.asarray(meta_test_o))], axis=1)
        
        
        
        
    #taking target wise mean
    meta_mean = meta_test_preds.mean(axis=1)
    ridge_mean = ridge_test_preds.mean(axis=1)
    enet_mean = enet_test_preds.mean(axis=1)
    svr_mean = svr_test_preds.mean(axis=1)
    
    meta_train_mean = meta_test_train.mean(axis=1)
    
    
    
    #saving test_predictions to dfs
    sub[target] = meta_mean
    ridge_test[target] = ridge_mean
    enet_test[target] = enet_mean
    svr_test[target] = svr_mean
    meta_train_df[target] = meta_train_mean