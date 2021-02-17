import pandas as pd
import re
import numpy as np


def cv_results_to_df(my_dict, cols_expand=('clf',)):
    df = pd.DataFrame(my_dict).drop('params', axis=1)
    expanded = []
    for col_expand in cols_expand:
        col_expanded = df.pop('param_' + col_expand + '__selected_model').apply(pd.Series)
        params_col_expanded = pd.DataFrame(col_expanded.loc[:, 1].tolist())
        col_expanded.drop(col_expanded.columns[1], inplace=True, axis=1)
        col_expanded.rename(columns={0: col_expand}, inplace=True)
        expanded.extend([params_col_expanded, col_expanded])

    sub_dfs = [df]
    sub_dfs.extend(expanded)
    result = pd.concat(sub_dfs, axis=1)
    result.columns = [re.sub("param.*__", "", col) for col in result.columns]
    return result


def remove_col_prefix(df):
    df.columns = [re.sub("param.*__", "", col) for col in df.columns]
    return df


def compute_means(df):
    prefixes = np.unique([re.split('[0-9]', col)[-1] for col in df.columns if re.search('[0-9]', col)])
    for prefix in prefixes:
        cols = [col for col in df.columns if prefix in col]
        df['mean' + prefix] = df[cols].aggregate('mean', axis=1)

    return df
