from sklearn import datasets
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.utils import resample
import pandas as pd

from .util import corrupt_gaussian, corrupt_snp


def get_openml_data(data_id, subsample_size=None, scaling='minmax', corrupt_type=None, noise_level=0.1, reshape=None):
    data = datasets.fetch_openml(data_id=data_id, as_frame=True)
    if scaling == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    label_enc = LabelEncoder()

    selector = (data['data'].dtypes == 'category')
    cat_columns = data['data'].loc[:, selector].columns
    x = pd.get_dummies(data['data'], columns=cat_columns, drop_first=True).to_numpy()

    y = data['target'].to_numpy()

    subsample_size = min(subsample_size, len(y)) if subsample_size else None
    x, y = resample(x, y, n_samples=subsample_size, stratify=y, replace=False)
    if corrupt_type:
        if corrupt_type == 'snp':
            x = corrupt_snp(x, noise_level=noise_level, pepper=True)
        elif corrupt_type == 'zero':
            x = corrupt_snp(x, noise_level=noise_level, pepper=False)
        else:
            x = corrupt_gaussian(x, noise_level=noise_level)

    if scaling:
        x = scaler.fit_transform(x)
    y = label_enc.fit_transform(y)

    if reshape:
        x = x.reshape(reshape)

    return x, y
