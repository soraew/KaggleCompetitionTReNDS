import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
import os
import sys

os.system("cp ../input/rapids/rapids.0.13.0 /opt/conda/envs/rapids.tar.gz")
os.system("cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null")
sys.path = ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib/python3.6"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path
os.system("cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/")

import cuml
import cudf


from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import ElasticNet, Ridge
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



#Data Loader
fnc_df = pd.read_csv("../input/trends-assessment-prediction/fnc.csv")
loading_df = pd.read_csv("../input/trends-assessment-prediction/loading.csv")

fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])
df = fnc_df.merge(loading_df, on="Id")


labels_df = pd.read_csv("../input/trends-assessment-prediction/train_scores.csv")
labels_df["is_train"] = True

df = df.merge(labels_df, on="Id", how="left")

test_df = df[df["is_train"] != True].copy()
train_df = df[df["is_train"] == True].copy()


# used for training SVR better because SVR is sensitive to scale.
# I initialy did not use this scale for trainig ridge and enet, but using this turned out to \
# have better cv for them too.
FNC_SCALE = 1/500

train_df[fnc_features] *= FNC_SCALE
test_df[fnc_features] *= FNC_SCALE

# excluded 'IC_20' features based on previous experiments using leave one out feature selection
loading_features.remove('IC_20')
features = fnc_features + loading_features

targets = ["age", "domain1_var1","domain1_var2", "domain2_var1", "domain2_var2"]