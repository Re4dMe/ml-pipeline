from kfp import dsl
from typing import NamedTuple


@dsl.component(
    # packages_to_install=["scikit-learn", "pandas"],
    base_image="dockeruser955/pipeline-container-image"  # "python:3.11",
)
def prepare_data(data_path: str):
    from sklearn import datasets
    import pandas as pd
    import time
    import datetime
    '''
    Access your own Data Server here.
    data = request('YourOwnDataServer')

    We use iris dataset as an example.
    '''

    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target

    df = df.dropna()
    df.to_csv(f'{data_path}/final_df.csv', index=False)
    df = pd.read_csv(f'{data_path}/final_df.csv')


@dsl.component(base_image="dockeruser955/pipeline-container-image")
def train_test_split(data_path: str):
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import numpy as np
    import datetime
    import os
    final_data = pd.read_csv(f'{data_path}/final_df.csv')

    target_column = 'species'
    x = final_data.loc[:, final_data.columns != target_column]
    y = final_data.loc[:, final_data.columns == target_column]

    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.2,
                                                        stratify=y,
                                                        random_state=37)
    np.save(f'{data_path}/x_train.npy', x_train)
    np.save(f'{data_path}/x_test.npy', x_test)
    np.save(f'{data_path}/y_train.npy', y_train)
    np.save(f'{data_path}/y_test.npy', y_test)


@dsl.component(base_image="dockeruser955/pipeline-container-image")
def detect_model_degradation(
    data_path: str,
    register_model_name: str,
    model_tag: str = 'Production',
    retrain_threshold: float = 0.7
) -> NamedTuple('outputs', retraining=bool, accuracy=float):  # type: ignore
    import numpy as np
    import mlflow
    from mlflow import MlflowClient
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    mlflow.set_tracking_uri("http://mlflow-tracking-server.kubeflow:5000")
    retrain_flag = False
    pipeline_outputs = NamedTuple('pipeline_outputs',
                                  retraining=bool,
                                  accuracy=float)
    try:
        register_model_uri = f"models:/{register_model_name}@{model_tag}"
        print(f'Load register model uri: {register_model_uri}')
        register_model = mlflow.sklearn.load_model(register_model_uri)
        print(f'Loaded successfully: {register_model_uri}')
    except Exception as err:
        # If no existing model, then trigger training pipeline
        print(f'Loaded failed. {err}')
        retrain_flag = True
        return pipeline_outputs(retraining=retrain_flag, accuracy=0)

    x_test = np.load(f'{data_path}/x_test.npy', allow_pickle=True)
    y_test = np.load(f'{data_path}/y_test.npy', allow_pickle=True)

    y_pred = register_model.predict(x_test)
    acc_score = accuracy_score(y_test, y_pred)

    print(f'accuracy: {acc_score}, retraining threshold: {retrain_threshold}')
    if acc_score < retrain_threshold:
        print('Trigger retraining pipeline')
        retrain_flag = True

    return pipeline_outputs(retraining=retrain_flag, accuracy=acc_score)


@dsl.component(base_image="dockeruser955/pipeline-container-image")
def test_data_prediction(data_path: str, model_info: dict):
    import numpy as np
    import os
    import mlflow

    artifact_path = model_info["artifact_path"]
    artifact_uri = model_info["artifact_uri"]
    print(f'artifact_path: {artifact_path}')
    print(f'artifact_uri: {artifact_uri}')

    mlflow.set_tracking_uri("http://mlflow-tracking-server.kubeflow:5000")
    model_uri = f"{artifact_uri}/{artifact_path}"
    logistic_reg_model = mlflow.sklearn.load_model(model_uri)

    X_test = np.load(f'{data_path}/x_test.npy', allow_pickle=True)
    y_pred = logistic_reg_model.predict(X_test)
    np.save(f'{data_path}/y_pred.npy', y_pred)

    X_test = np.load(f'{data_path}/x_test.npy', allow_pickle=True)
    y_pred_prob = logistic_reg_model.predict_proba(X_test)
    np.save(f'{data_path}/y_pred_prob.npy', y_pred_prob)

    return model_uri
