FROM rocm/pytorch:rocm5.2.3_ubuntu20.04_py3.7_pytorch_1.10.0

ARG USER
ARG PASS

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8


ADD . /workspace/unet3d
WORKDIR /workspace/unet3d

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
RUN apt-get install -y vim

RUN pip install --upgrade pip
RUN pip install --disable-pip-version-check -r requirements.txt
