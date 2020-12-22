from sklearn import datasets
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.utils import resample
import pandas as pd


def get_openml_data(data_id, subsample_size=None, scale='standard'):
    data = datasets.fetch_openml(data_id=data_id, as_frame=True)
    if scale == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    label_enc = LabelEncoder()

    selector = (data['data'].dtypes == 'category')
    cat_columns = data['data'].loc[:, selector].columns
    x = pd.get_dummies(data['data'], columns=cat_columns, drop_first=True)
    y = data['target']

    if subsample_size:
        x, y = resample(x, y, n_samples=subsample_size, stratify=y)

    x = scaler.fit_transform(x)
    y = label_enc.fit_transform(y)

    return x, y