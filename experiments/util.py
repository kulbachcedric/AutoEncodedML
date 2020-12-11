import pandas as pd
import re


def cv_results_to_df(my_dict, column_clf='clf'):
    df = pd.DataFrame(my_dict).drop('params', axis=1)
    df_clf = df.pop('param_' + column_clf + '__selected_model').apply(pd.Series)
    clf_params = pd.DataFrame(df_clf.loc[:, 1].tolist())
    df_clf.drop(df_clf.columns[1], inplace=True, axis=1)
    df_clf.columns = [column_clf]
    result = pd.concat([df, df_clf, clf_params], axis=1)
    result.columns = [re.sub("param.*__", "", col) for col in result.columns]
    return result
