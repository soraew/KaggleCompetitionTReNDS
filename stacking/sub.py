from loading import test_df
from stacking import sub

from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import ElasticNet, Ridge
import numpy as np
import pandas as pd

if __name__ == "__main__":

    #submit function for saving predictions to submit format csv
    def submit(pred_df, test_df):
        def to_sub(pred_df):
            sub_df = pd.melt(pred_df[["Id", "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]], id_vars=['Id'], value_name='Predicted')

            sub_df["Id"] = sub_df["Id"].astype("str") + "_" +  sub_df["variable"].astype("str")

            sub_df = sub_df.drop("variable", axis=1).sort_values("Id")

            #assert here is for debugging
            assert sub_df.shape[0] == test_df.shape[0]*5

            return sub_df

        test_df.reset_index(drop=True, inplace=True)
        pred_df.reset_index(drop=True, inplace=True)
        pred_df['Id'] = test_df['Id'].astype('int')
        sub_df = to_sub(pred_df)
        sub_df.to_csv('sub.csv', index=False)
        
    # the submission csv will be saved as "sub.csv"a
    submit(sub, test_df)