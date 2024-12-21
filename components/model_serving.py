from kfp import dsl


@dsl.component(base_image="dockeruser955/pipeline-container-image")
def model_serving(tracking_uri: str,
                  model_uri: str,
                  pvc_name: str):
    from kubernetes import client
    from kserve import (KServeClient, constants, utils,
                        V1beta1InferenceService, V1beta1InferenceServiceSpec,
                        V1beta1PredictorSpec, V1beta1ModelSpec,
                        V1beta1ModelFormat, V1beta1SKLearnSpec)
    import time
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)

    namespace = 'kubeflow'  # utils.get_default_target_namespace()
    name = 'sklearn-iris'
    kserve_version = 'v1beta1'
    api_version = constants.KSERVE_GROUP + '/' + kserve_version

    model_uri = model_uri.split('backend')[1]
    storage_uri = f'pvc://{pvc_name}{model_uri}'

    # if the model is store in a remote storage uri, use
    # storage_uri = s3://your-model-uri or gs://your-model-uri
    print(f'storage_uri: {storage_uri}')

    protocol_version = "v2"
    # Enable Prometheus
    annotations = {
        'serving.kserve.io/enable-prometheus-scraping': 'True'
    }
    predictor = V1beta1PredictorSpec(
        min_replicas=1,
        model=V1beta1ModelSpec(
            model_format=V1beta1ModelFormat(name="mlflow", ),
            storage_uri=storage_uri,
            protocol_version=protocol_version,
        ),
    )

    isvc = V1beta1InferenceService(
        api_version=constants.KSERVE_V1BETA1,
        kind=constants.KSERVE_KIND,
        metadata=client.V1ObjectMeta(name=name,
                                     namespace=namespace,
                                     annotations=annotations),
        spec=V1beta1InferenceServiceSpec(predictor=predictor),
    )

    KServe = KServeClient()

    try:
        KServe.delete(name=name, namespace=namespace)
        print("Delete existing inference-service.")
    except Exception as e:
        print(f"Delete existing inference-service failed. Exception: {e}")
    time.sleep(10)

    KServe.create(isvc)
