from loading import *

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import numpy as np

n_splits = 5

features = fnc_features + selected2

from sklearn.model_selection import train_test_split

targets= ["age","domain1_var1","domain1_var2", "domain2_var1", "domain2_var2"]

ridge_df = pd.DataFrame(df.copy())


kf = KFold(n_splits = n_splits)

sub_targets_container=[]

for alpha in [0.001, 0.003, 0.005]:
    
    #this is for the test predictions
    sub_targets = pd.DataFrame()

    for target in targets:
        
        ##here, we use ridge_df for trainig, and test_df for submission
        ridge_df = ridge_df.loc[ridge_df[target].notnull()]
        
        sub_cv = np.zeros((test_df.shape[0], n_splits))
        sub_cv = pd.DataFrame(sub_cv)


        target_gain = 0
        

        
        for fold, (train_ind, val_ind) in enumerate(kf.split(ridge_df)):
            
            X_train, X_val = ridge_df.iloc[train_ind][features], ridge_df.iloc[val_ind][features]
            y_train, y_val =  ridge_df.iloc[train_ind][target], ridge_df.iloc[val_ind][target]
            
            
            ridge = Ridge(alpha=alpha)
            
            ridge.fit(X_train, y_train)
            predicted = ridge.predict(X_val)

            loss = metric(y_val, predicted)

            
            gain = round(float(svm_loss_dict[target])-float(loss), 4)
            target_gain += gain
            
            

            ridge_sub = Ridge(alpha=alpha)
            ridge_sub.fit(X_train, y_train)
            sub_preds = ridge_sub.predict(test_df[features])
            
            
            sub_cv.iloc[:, fold] = sub_preds#this should work if predictions are correct
            
            
        mean = sub_cv.mean(axis=1)#this is correct
            
        
        sub_targets['{}_pred_{}'.format(str(alpha), target)] = mean


        target_gain /= n_splits
        target_gain = round(target_gain, 4)
        
            

        print('\nTARGET = ',target)
        print('ALPHA=',alpha)
        print('TARGET_GAIN            =         ',target_gain)
        
        
    sub_targets_container.append(sub_targets)

print('\n\nDONE')
