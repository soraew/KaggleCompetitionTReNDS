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

#can only be run in a directory in kaggle using GPU
root = '/kaggle/input/trends-assessment-prediction/'
fnc_df = pd.read_csv('/kaggle/input/trends-assessment-prediction/fnc.csv')
loading_df = pd.read_csv(root+'loading.csv')
reveal = pd.read_csv(root+'reveal_ID_site2.csv')
numbers = pd.read_csv(root+'ICN_numbers.csv')
sample_sub = pd.read_csv(root+'sample_submission.csv')
labels_df = pd.read_csv(root+'train_scores.csv')

df=pd.DataFrame()
fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])

df = fnc_df.merge(loading_df, on="Id")

labels_df['is_train'] = True

df = df.merge(labels_df, on="Id", how="left")


test_df = df[df["is_train"] != True].copy()
df = df[df["is_train"] == True].copy()

def metric(y_true, y_pred):
    return np.mean(np.sum(np.abs(y_true - y_pred), axis=0)/np.sum(y_true, axis=0))

df[fnc_features] *= 1.0/500.0
test_df[fnc_features] *= 1.0/500.0

svm_loss_dict={'age': '0.1445',
 'domain1_var1': '0.1512',
 'domain1_var2': '0.1512',
 'domain2_var1': '0.1818',
 'domain2_var2': '0.1761'}

#finally selected ICs for submission (removed some after calculating feature importance using lofo-importance)
selected2 =  ['IC_01', 'IC_02', 'IC_03', 'IC_04', 'IC_06', 'IC_07', 'IC_08',
       'IC_09', 'IC_10', 'IC_11', 'IC_12', 'IC_13', 'IC_14', 'IC_15',
       'IC_16', 'IC_17', 'IC_18', 'IC_21', 'IC_22', 'IC_24', 'IC_26',
       'IC_28', 'IC_29']

domains = ["age","domain1_var1","domain1_var2", "domain2_var1", "domain2_var2"]