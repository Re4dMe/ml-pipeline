## TL;DR
This repo demonstrate how to deploy ML pipeline by using `Kubeflow`, `MLFlow`, `KServe` and `Prometheus` to achieve automatic data monitoring, model training, model serving and service monitoring, then expose the inference services by `Istio`.

## About

In this work, a ML pipeline is set up and consist of 6 stages. Each stage is executed by a K8s pod.
The stages are as follows:
<ol>
<li>
Data ingestion: 

Collecting data into the system for processing. Data can come from various sources such as databases, APIs, or flat files like CSV or JSON. We use sklearn iris dataset and assume it is sourced from an external data server. The data will be store in a `Persistent Volume`.
</li>
<li>Train test split:

The data will be split into training set and test set.
</li>
<li> Model degradation detection: 

Over time, machine learning models can degrade in performance due to changes in the input data or the environment (e.g., concept drift). In this stage, the model is evaluated, and if its performance is above a certain threshold (the model is already performing well enough), the next stage will not be triggered. Otherwise, the pipeline will proceed to the Model Training stage.
</li>
<li> Model training/evaluation:

In this stage, the model is trained using the training dataset. Training parameters and results will be saved using `MLflow`. If the training result is better than the currently deployed model, the next stage, Model Registration, is triggered.
</li>
<li> Model Registration:

The model is registered in the `MLflow` model registry and labeled as the new serving model, ready for deployment.
</li>

<li> Model Serving:

The model is served using KServe. An InferenceService will be created and exposed to public access via an `Istio` gateway.
</li>
</ol>
Please note that this pipeline may not be the optimal setup for a production scenario. It is designed primarily for demonstration purposes to showcase the different stages of a machine learning pipeline. In a production environment, additional considerations such as performance optimization, error handling, and continuous monitoring should be implemented.


## Requirement
- Kubeflow pipeline
- MLflow
- KServe
- Docker

## Deploy the pipeline

### Step 1: Deploy Kubeflow Pipelines and MLflow Tracking Services
Deploy the Kubeflow Pipelines
```
export PIPELINE_VERSION=2.3.0
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=$PIPELINE_VERSION"
```

Apply the configurations for MLflow tracking services and pods
```
kubectl apply -f k8s-yaml\tracking-server\tracking-server-pod.yaml
kubectl apply -f k8s-yaml\tracking-server\tracking-server-services.yaml
```

Apply the configurations for pvc used by tracking services
```
kubectl apply -f k8s-yaml\persistent-volume-claim\tracking-pvc.yaml
kubectl apply -f k8s-yaml\persistent-volume-claim\ml-data-pvc.yaml
```
### Step 2: Expose services in clusters
To expose the services, *Gateway* and *VirtualService* *CRD* provide by `Istio` are needed.
However for simplicity, the kfp UI and mlflow server are exposed to local by forwarding the port of services:

Kubeflow UI
```
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```
MLflow tracking server
```
kubectl port-forward -n kubeflow svc/mlflow-tracking-server 5000:5000
```

Port-forwarding *istio-ingressgateway* is *Not* required unless *External IP* of `LoadBalancer` can not be obtain. 
```
kubectl port-forward -n istio-system service/istio-ingressgateway 8081:80
```
Instead, the *InferenceService* will be exposed to public by `istio-ingressgateway`, so they can be access from outside by *external IP*.
`DNS` might be set up to allow access by Domain name.
### Step 3: Edit Role of pipeline-runner [[2]](#2)
By default, the KServe will use user pipeline-runner in kubeflow namespace to create inference service. However, the user is not in the api groups `serving.kserve.io` by default. Thus we have to add the permission by the following command:

Create role & edit role of pipeline-runner
```
kubectl apply -f k8s-yaml\roles\role.yaml
kubectl apply -f k8s-yaml\roles\role-binding.yaml
kubectl edit role -n kubeflow pipeline-runner
```

Then add the api groups in the yaml config
```
- apiGroups:
  - serving.kserve.io
  resources:
  - inferenceservices
  verbs:
  - get
  - list
  - watch
  - create
  - update
  - patch
  - delete
```
Check authentication 
```
kubectl auth can-i create InferenceServices --as=system:serviceaccount:kubeflow:pipeline-runner -n kubeflow
```
The reponse should be *yes*.

### Step 4: Configuring the default domain for all Knative Services on a cluster
Edit config-domain
```
kubectl edit configmap config-domain -n knative-serving
```
Replace `data:` section with new domain name (`knative.dev` as an example)
```
knative.dev: ""
```
To make this domain public accessible, see [[4]](#4).

### Step 5: Start pipeline daemon process
Start process that monitor for new data, and trigger re-training pipeline when model degradation detected. If a new model is better than the one currently running, the pipeline automatically deploy it to KServe (Inference Platform.)

```
python pipeline.py
```

### Step 6: Check the status of pipelines and experiments
Head to [https://127.0.0.1:8080/#/runs ](http://127.0.0.1:8080/#/runs) to check status of pipelines, and [https://127.0.0.1:5000/#/runs ](http://127.0.0.1:5000/#/runs) to check mlflow experiments/models. Or they might be access by domain name if a DNS has been set up.

### Step 7: Request inference-service
Prepare test data [[3]](#3) (See `iris-input-v1.json` and `iris-input-v2.json`)

Check the `External IP` by issue:
```
export INGRESS_NAME=istio-ingressgateway
export INGRESS_NS=istio-system
kubectl get svc "$INGRESS_NAME" -n "$INGRESS_NS"
```
Perform inference
```
export INGRESS_HOST=$(kubectl -n "$INGRESS_NS" get service "$INGRESS_NAME" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
export INGRESS_PORT=$(kubectl -n "$INGRESS_NS" get service "$INGRESS_NAME" -o jsonpath='{.spec.ports[?(@.name=="http2")].port}')
export SERVICE_HOSTNAME=$(kubectl get inferenceservice sklearn-iris -n kubeflow -o jsonpath='{.status.url}' | cut -d "/" -f 3)
curl -v -H "Host: ${SERVICE_HOSTNAME}" -H "Content-Type: application/json" http://${INGRESS_HOST}:${INGRESS_PORT}/v2/models/sklearn-iris/infer -d @./iris
-input-v2.json
```

### Step 8: Monitor inference service with Prometheus
Exposing a `Prometheus` metrics port and request
```
kubectl port-forward -n <namespace> <pod-name> <desired-port>:8082
curl -v -H "Content-Type: application/json" <ip-of-service>:<desired-port>/metrics
```

## Pod/Service/Deployment management

### kubectl delete pods/service:
Example to delete pods with *pod-name* as name. Awk '{ print $1 }' to filter out the name of the pod without other information.
```
kubectl get pods -n <namespace> | awk '{ print $1 }' | grep <pod-name> | xargs kubectl delete pods -n <namespace>
```

### kubectl delete pvc:

If the process stuck at pvc status: terminating,
then type:
```
kubectl edit pv -n kubeflow *pv-name* 
```
(get pv-name by kubectl get pv -n <namespace>, lookup the CLAIM column)
and delete 'finalizers' content:
```
- kubernetes.io/pv-protection
```

### Apply inference-service config manually
```
inference-service.yaml:

apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "sklearn-iris"
spec:
  predictor:
    model:
      modelFormat:
        name: sklearn
      storageUri: "gs://kfserving-examples/models/sklearn/1.0/model"

kubectl apply -n {namespace} -f {path-to-inference-service-yaml}
kubectl port-forward -n {namespace} service/istio-ingressgateway {forwarding-port}:80
kubectl get inferenceservice -A
curl -v -H "Host: {url-of-service}" -H "Content-Type: application/json" "http://localhost:{forwarding-port}/v1/models/{inference-model-name}:predict" -d @{path-to-input}
```

### Deploy deployment and service [[1]](#1)

Create yaml for deployment and service. Start the deployments and services in cluster by:
```
kubectl apply -f <path-to-yaml-file(-directory)> 
```
Now we're allow to see info by typing:
kubectl get service -n *namespace* 

### Edit containerRuntimeExecutor in configmap 
```
kubectl edit -n kubeflow configmap workflow-controller-configmap
```

### Build docker file for pipeline component

```
cd <directory-of-container-env-dockerfile>
docker build -t <name-of-image> .
```
and then
```
docker tag <image-name> <tag-image-name>
docker login
docker push <image-name>
```

## Reference:

<a id="1">[1]</a> https://aahansingh.com/mlflow-on-kubernetes

<a id="2">[2]</a> https://stackoverflow.com/questions/64551602/forbidden-systemserviceaccountdefaultdefault-cannot-create-resource-how-t

<a id="3">[3]</a> https://mlflow.org/docs/latest/deployment/deploy-model-to-kubernetes/tutorial.html

<a id="4">[4]</a> https://knative.dev/docs/serving/using-a-custom-domain/#verification-steps