FROM rocm/pytorch:rocm5.2.3_ubuntu20.04_py3.7_pytorch_1.12.1

ARG USER
ARG PASS

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN mkdir -p /src

WORKDIR /src

RUN git clone https://${USER}:${PASS}@github.com/AMD-ACP/MLPerf -b dev && \
    cd /src/MLPerf/ImageClassification/resnet50/pytorch && \
    bash requirements.sh && \
    cd /src/MLPerf/ && rm -r .git


