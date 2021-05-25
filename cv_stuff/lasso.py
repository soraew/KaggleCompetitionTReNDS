from sklearn.linear_model import Lasso
from cv_and_sub import cv_and_sub
from loading import *


alphas = [3e-5,1e-4, 0.001, 0.003,  0.01,  0.03,  0.1]
targets = domains
features = fnc_features + selected2

sub_targets_container = {}

workeds = []
for alpha in alphas:
    model = Lasso(alpha=alpha)

    sub_targets, worked = cv_and_sub(train = df,
                test = test_df,
                model = model,
                alpha = alpha,
                targets = domains,
                features = fnc_features + selected2)
    
    
    
    sub_targets_container.update({alpha:sub_targets})
    workeds.append('\n'.join(worked))
    for_use = ''.join(workeds)