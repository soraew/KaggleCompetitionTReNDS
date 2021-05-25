from loading import *
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import ElasticNet, Ridge
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



#function for quickly testing cv and checking loss
def cv_test(model, train_df=train_df, test_df=test_df, features=features, targets=targets, folds=5):


    kf = KFold(n_splits = folds)

    losses = []


    test_preds = pd.DataFrame()

    for target in targets:

        print(target)

        test_pred_cv = pd.DataFrame()
        loss = 0

        train = train_df.loc[train_df[target].notnull()]
        for fold, (train_ind, val_ind) in enumerate(kf.split(train)):
                
            
            X_train, X_val = train.iloc[train_ind][features], train.iloc[val_ind][features]
            y_train, y_val = train.iloc[train_ind][target], train.iloc[val_ind][target]

            
            try:
                model.fit(X_train, y_train)#model がsklearn のものだった場合
                pred = model.predict(X_val)
                test_pred = model.predict(test_df[features])
            except:
                if target == 'age':
                    c = 100
                else:
                    c = 10
                model = SVR(C=c, cache_size=3000.0)
                model.fit(cd.DataFrame(X_train), cd.Series(y_train))
                pred = model.predict(cd.DataFrame(X_val))
                pred = np.asarray(pred)
                test_pred = np.asarray(model.predict(cd.DataFrame(test_df[features])))
                

            
            loss+= metric(y_val, pred)/folds

                
            test_pred_cv = pd.concat([test_pred_cv, pd.Series(test_pred, name=f'{fold}')], axis=1)
        test_mean = test_pred_cv.mean(axis=1)
        test_preds[target] = test_mean

        
        
        losses.append(loss)
        print(loss, '\n\n')
    
    
    final_score = losses[0]*0.30 + losses[1]*0.175 + losses[2]*0.175+ losses[3]*0.175+ losses[4]*0.175
    print(final_score)
    return(test_preds, losses)