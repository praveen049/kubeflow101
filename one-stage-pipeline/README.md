# kubeflow101

This package is for deplouyment of a one stage ML pipeline. The ML pipeline is a trivial implementation of LinearRegression model.

The model is trained offline using Salary Data using sklearn LinearRegression and model is stored locally.

The inference implementation load the local model and exposes a Flask REST API which can take input and use the model to generate prediction and return it.

The Docker package include the ML code and a Dockerfile for packaging the code as a container.

The docker can be build manually with the below commands

docker build -t inf:latest .
docker tag <imageid> praveen049/latest
docker push praveen049/inf

Alternatively, build_image.sh can be executed to automativally build, tag and upload the image.

Kubernetes directory contains the pod and service yaml description to deploy the Inference pipeline on a kubernetes cluster.

Below are the instructions for deploying the pipeline on kubernetes and exposing the REST API end point as NodePort

kubectl create -f infer.yml
kubectl create -f infer-svc.yml


Kubeflow directory contains the implementation for deploying the inference as Kubeflow pipeline # kubeflow101

