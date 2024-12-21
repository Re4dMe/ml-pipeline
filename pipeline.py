from time import sleep
from kfp import dsl
from kfp import compiler
import kfp
import requests
from kfp import kubernetes
from components.data_ingestion import (detect_model_degradation, prepare_data,
                                       train_test_split)
from components.model_monitor import model_register, model_training
from components.model_serving import model_serving
from argparse import ArgumentParser
import requests


@dsl.pipeline(name="iris-pipeline")
def ml_pipeline(
    data_path: str,
    config_path: str,
):
    register_model_name = "SklearnLogisticRegression"
    tracking_uri = 'http://mlflow-tracking-server.kubeflow:5000'
    mlflow_pvc_name = 'mlflow-pvc'
    ml_data_pvc_name = 'ml-data-pvc'
    mount_data_path = '/data'
    mount_mlflow_path = '/opt/mlflow/backend'
    # Load Kubernetes configuration
    # kubernetes.Config.load_kube_config()

    # Create PVC dynamically
    '''
    pvc = kubernetes.CreatePVC(
        # can also use pvc_name instead of pvc_name_suffix to use a pre-existing PVC
        pvc_name_suffix='-iris-mlflow-pvc',
        access_modes=['ReadWriteOnce'],
        size='5Mi',
        storage_class_name='standard')
    '''

    # Data preparation
    prepare_data_task = prepare_data(data_path=data_path)
    prepare_data_task.set_caching_options(False)
    kubernetes.mount_pvc(
        prepare_data_task,
        pvc_name=ml_data_pvc_name,  # pvc.outputs['name']
        mount_path=mount_data_path)
    # prepare_data_task.after(pvc)

    # Split data into Train and Test set
    train_test_split_task = train_test_split(data_path=data_path)
    train_test_split_task.set_caching_options(False)
    kubernetes.mount_pvc(train_test_split_task,
                         pvc_name=ml_data_pvc_name,
                         mount_path=mount_data_path)
    train_test_split_task.after(prepare_data_task)

    # Model degration detection
    detect_model_degradation_task = detect_model_degradation(
        data_path=data_path, register_model_name=register_model_name)
    detect_model_degradation_task.set_caching_options(False)
    kubernetes.mount_pvc(detect_model_degradation_task,
                         pvc_name=ml_data_pvc_name,
                         mount_path=mount_data_path)
    kubernetes.mount_pvc(detect_model_degradation_task,
                         pvc_name=mlflow_pvc_name,
                         mount_path=mount_mlflow_path)

    detect_model_degradation_task.after(train_test_split_task)

    with dsl.If(detect_model_degradation_task.outputs['retraining'] == True,
                name='retraining-condition'):
        # Model training
        model_training_task = model_training(
            tracking_uri=tracking_uri,
            data_path=data_path,
            best_acc=detect_model_degradation_task.outputs['accuracy'])
        model_training_task.set_caching_options(False)
        kubernetes.mount_pvc(model_training_task,
                             pvc_name=ml_data_pvc_name,
                             mount_path=mount_data_path)
        kubernetes.mount_pvc(model_training_task,
                             pvc_name=mlflow_pvc_name,
                             mount_path=mount_mlflow_path)
        model_training_task.after(detect_model_degradation_task)

        with dsl.If(model_training_task.outputs['model_register_flag'] == True,
                    name='model-register-condition'):
            # Model register
            model_register_task = model_register(
                tracking_uri=tracking_uri,
                runs_model_uri=model_training_task.outputs['runs_model_uri'],
                register_model_name=register_model_name,
                model_tag='Production')
            model_register_task.set_caching_options(False)
            kubernetes.mount_pvc(model_register_task,
                                 pvc_name=mlflow_pvc_name,
                                 mount_path=mount_mlflow_path)
            model_register_task.after(model_training_task)

            # Model deployment
            model_serving_task = model_serving(
                tracking_uri=tracking_uri,
                model_uri=model_training_task.outputs['model_uri'],
                pvc_name=mlflow_pvc_name)
            model_serving_task.set_caching_options(False)
            kubernetes.mount_pvc(model_serving_task,
                                 pvc_name=mlflow_pvc_name,
                                 mount_path=mount_mlflow_path)
            model_serving_task.after(model_register_task)
    '''
    delete_pvc = kubernetes.DeletePVC(
        pvc_name=pvc.outputs['name']).after(model_serving_task)
    delete_pvc.set_caching_options(False)
    '''


def single_run_mode():
    run = client.create_run_from_pipeline_package(
        'pipeline.yaml',
        arguments={
            'data_path': 'data/',
            'config_path': 'configs/',
        },
        enable_caching=None,
    )


def continuous_mode():
    PipelineTask = True
    while PipelineTask:
        '''
        Monitor for new data continuously.
        When the amount of data exceeds a threshold, the remote endpoint should return a signal,
        then the re-training pipeline would be trigger.
        '''

        r = requests.get('https://DataServer/IsDataAvailableEndPoint')

        if 'ready' in r.text:
            run = client.create_run_from_pipeline_package(
                'pipeline.yaml',
                arguments={
                    'data_path': 'data/',
                    'config_path': 'config/',
                },
                enable_caching=True,
            )
        print('Pipeline completed.')
        sleep(3)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--mode", help="running mode", default="single")
    args = parser.parse_args()

    compiler.Compiler().compile(ml_pipeline, 'pipeline.yaml')
    client = kfp.Client(host="http://127.0.0.1:8080/")
    print(client.list_experiments())

    if args.mode == 'single':
        single_run_mode()
    elif args.mode == 'continuous':
        continuous_mode()
    else:
        single_run_mode()
