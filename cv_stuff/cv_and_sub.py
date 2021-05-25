from loading import *

from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

#this is only for models with alphas


def cv_and_sub(train, test, model, alpha, targets, features, folds=5):
    

    worked = []#this is for showing the alpha, target pairs that worked better than svm
    
    kf = KFold(n_splits=folds)
    n_splits = folds
    
    features = features
    targets = targets
    alpha = alpha
    
    test = test.copy()
    train = train.copy()

    #this is for the test predictions
    sub_targets = pd.DataFrame()

    for target in targets:

        ##here, we use train for trainig, and test for submission
        train = train.loc[train[target].notnull()]

        sub_cv = np.zeros((test.shape[0], n_splits))
        sub_cv = pd.DataFrame(sub_cv)


        target_gain = 0



        for fold, (train_ind, val_ind) in enumerate(kf.split(train)):

            X_train, X_val = train.iloc[train_ind][features], train.iloc[val_ind][features]
            y_train, y_val =  train.iloc[train_ind][target], train.iloc[val_ind][target]


            model.fit(X_train, y_train)
            predicted = model.predict(X_val)

            loss = metric(y_val, predicted)


            gain = round(float(svm_loss_dict[target])-float(loss), 4)
            target_gain += gain


            model_sub = model
            model_sub.fit(X_train, y_train)
            sub_preds = model_sub.predict(test[features])


            sub_cv.iloc[:, fold] = sub_preds#this should work if predictions are correct


        mean = sub_cv.mean(axis=1)#this is correct


        sub_targets['{}_pred_{}'.format(str(alpha), target)] = mean


        target_gain /= n_splits
        target_gain = round(target_gain, 4)



        print('\nTARGET = ',target)
        print('ALPHA=',alpha)
        print('TARGET_GAIN            =         ',target_gain)
        
        
        if target_gain > 0:
            worked.append('alpha : {} target : {}'.format(str(alpha), target))



    print('\n\nDONE')
    
    
    

    return (sub_targets, worked)
