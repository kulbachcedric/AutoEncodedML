from pipelinehelper import PipelineHelper
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from auto_encoder.ae import Autoencoder
from auto_encoder.sklearn_wrapper import AETransformer
from dataloader.openml import get_openml_data
from dataloader.utils import get_train_test_indices


if __name__ == '__main__':

    x, y = get_openml_data(44)
    split_indices = get_train_test_indices(x)

    dataset_ids = [40996, 40668, 1492, 44]
    results = []

    for dataset_id in dataset_ids:
        Autoencoder.cache_clear()
        x, y = get_openml_data(dataset_id)
        split_indices = get_train_test_indices(y, test_size=0.2)



        pipe = Pipeline([
            ('ae', AETransformer(input_shape=x[0].shape)),
            ('clf', PipelineHelper([('svm', SVC(max_iter=500)),
                                    ('log_reg', LogisticRegression(max_iter=500)),
                                    ('xgb', XGBClassifier())
                                    ]))
        ])

        params = {
            'ae__first_dim': [25, 50],
            'ae__latent_dim': [10, 20],
            'ae__n_layers': [3],
            'clf__selected_model': pipe.named_steps['clf'].generate({
                'svm__C': [1, 10],
            })
        }

        grid = GridSearchCV(pipe, param_grid=params, cv=[split_indices])
        grid.fit(x, y)


