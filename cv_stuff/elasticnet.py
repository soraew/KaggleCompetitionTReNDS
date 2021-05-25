from loading import *

from sklearn.linear_model import ElasticNet
from cv_and_sub import cv_and_sub


alphas = [1e-4, 0.001, 0.003, 0.005]
ratios = [0,0.25, 0.5, 0.75, 1.0]


targets = domains
features = fnc_features + selected2

sub_targets_container = {}

workedss = []

for ratio in ratios:
    
    workeds = []
    
    for alpha in alphas:
        
        model = ElasticNet(alpha=alpha, l1_ratio=ratio, max_iter=5000)

        sub_targets, worked = cv_and_sub(train = df,
                    test = test_df,
                    model = model,
                    alpha = alpha,
                    targets = domains,
                    features = fnc_features + selected2)



        sub_targets_container.update({alpha:sub_targets})
        workeds.append('\n'.join(worked))
        
    workedss.append(''.join(workeds))
    
    use = ''.join(workedss)