ARG FROM_IMAGE_NAME=compute-artifactory.amd.com:5000/rocm-plus-docker/framework/compute-rocm-rel-5.1:57_ubuntu18.04_py3.7_pytorch_release-1.10_ed9b160_40
FROM ${FROM_IMAGE_NAME}

ADD . /workspace/unet3d
WORKDIR /workspace/unet3d

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
RUN apt-get install -y vim

RUN pip install --upgrade pip
RUN pip install --disable-pip-version-check -r requirements.txt
