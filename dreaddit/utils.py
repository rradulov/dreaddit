import pandas as pd
import numpy as np

from sklearn.feature_selection import mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor


def calc_MI(df_clean, y):

    features = df_clean.drop(columns=['confidence'])
    MI = mutual_info_classif(features, y)
    headers = features.columns
    MI_vars = pd.Series(index=headers, data=MI).sort_values(ascending=False)
    MI_vars_selected = MI_vars[MI_vars > 0]

    df_post_MI = df_clean[MI_vars_selected.index.to_list()]

    return df_post_MI


def calc_vif(df_post_MI):
    vif = pd.DataFrame()
    vif['variables'] = df_post_MI.columns
    vif['VIF'] = [
        variance_inflation_factor(df_post_MI, i)
        for i in range(df_post_MI.shape[1])
    ]

    VIF_df = vif[vif['VIF'] < 30].reset_index(drop=True)

    return VIF_df['variables'].to_list()
