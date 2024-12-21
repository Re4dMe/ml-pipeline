from kfp import dsl
from typing import NamedTuple


@dsl.component(base_image="dockeruser955/pipeline-container-image")
def model_training(
    tracking_uri: str,
    data_path: str,
    best_acc: float,
    aws_access_key_id: str = "",
    aws_secret_access_key: str = "",
    aws_default_region: str = ""
) -> NamedTuple('outputs',
                artifact_path=str,
                artifact_uri=str,
                model_uri=str,
                runs_model_uri=str,
                model_register_flag=bool):  # type: ignore
    import numpy as np
    import pickle
    import mlflow
    import datetime
    from mlflow.models import infer_signature
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from mlflow import MlflowClient

    pipeline_outputs = NamedTuple('pipeline_outputs',
                                  artifact_path=str,
                                  artifact_uri=str,
                                  model_uri=str,
                                  runs_model_uri=str,
                                  model_register_flag=bool)
    now = datetime.datetime.now()

    # Set AWS credentials in the environment
    # os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
    # os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
    # os.environ["AWS_DEFAULT_REGION"] = aws_default_region

    # log and register the model using MLflow scikit-learn API
    mlflow.set_tracking_uri(tracking_uri)

    experiment_id = mlflow.create_experiment(
        f"test-{now.strftime('%Y-%m-%d-%H-%M-%S')}")

    artifact_path = "MLmodel"

    retrain_accuracy = 0
    with mlflow.start_run(experiment_id=experiment_id) as run:
        mlflow.log_param('max_iter', 100)
        x_train = np.load(f'{data_path}/x_train.npy', allow_pickle=True)
        y_train = np.load(f'{data_path}/y_train.npy', allow_pickle=True)
        classifier = LogisticRegression(max_iter=100)
        classifier.fit(x_train, y_train)
        score = classifier.score(x_train, y_train)

        with open(f'{data_path}/model.pkl', 'wb') as f:
            pickle.dump(classifier, f)

        # Infer the model signature
        x_test = np.load(f'{data_path}/x_test.npy', allow_pickle=True)
        y_test = np.load(f'{data_path}/y_test.npy', allow_pickle=True)

        y_pred = classifier.predict(x_test)
        signature = infer_signature(x_test, y_pred)

        retrain_accuracy = accuracy_score(y_test, y_pred)

        # Log metric
        mlflow.log_metric("score", score)

        # Log model artifact
        mlflow.log_artifact(local_path=f'{data_path}/model.pkl',
                            artifact_path=artifact_path)
        mlflow.log_artifact(local_path=f'{data_path}/y_test.npy',
                            artifact_path=artifact_path)
        mlflow.log_artifact(local_path=f'{data_path}/x_test.npy',
                            artifact_path=artifact_path)

        # Log model
        model_info = mlflow.sklearn.log_model(
            sk_model=classifier,
            artifact_path=artifact_path,
            signature=signature,
        )

    model_uri = f"{run.info.artifact_uri}/{artifact_path}"
    runs_model_uri = f"runs:/{run.info.run_id}/{artifact_path}"
    model_register_flag = False

    # If the performance is improved, register model linked to artifact location
    if retrain_accuracy > best_acc:
        model_register_flag = True

    return pipeline_outputs(artifact_path=artifact_path,
                            artifact_uri=run.info.artifact_uri,
                            model_uri=model_uri,
                            runs_model_uri=runs_model_uri,
                            model_register_flag=model_register_flag)


@dsl.component(base_image="dockeruser955/pipeline-container-image")
def model_register(tracking_uri: str,
                   runs_model_uri: str,
                   register_model_name: str,
                   model_tag: str = 'Production'):
    import numpy as np
    import mlflow
    import datetime
    from mlflow.models import infer_signature
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from mlflow import MlflowClient

    def print_model_info(rm):
        print("--Model--")
        print("name: {}".format(rm.name))
        print("aliases: {}".format(rm.aliases))

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    model_version = mlflow.register_model(runs_model_uri, register_model_name)
    client.set_registered_model_alias(register_model_name, model_tag,
                                      model_version.version)
    model = client.get_registered_model(register_model_name)
    print_model_info(model)
