FROM rocm/pytorch:rocm5.2.3_ubuntu20.04_py3.7_pytorch_1.12.1

ENV WORKSPACE_DIR=/workspace
RUN mkdir -p $WORKSPACE_DIR
WORKDIR $WORKSPACE_DIR

ENV DATASET_DIR=/data
RUN mkdir -p $DATASET_DIR

RUN pip install --upgrade pip cmake

RUN apt update
RUN apt autoremove -y
RUN apt install -y sox ninja-build

COPY requirements.txt $WORKSPACE_DIR
COPY torchaudio-0.13.0a0+597a735-cp37-cp37m-linux_x86_64.whl $WORKSPACE_DIR
COPY apex-0.1-cp37-cp37m-linux_x86_64.whl $WORKSPACE_DIR
COPY run.sh $WORKSPACE_DIR
COPY scripts $WORKSPACE_DIR/scripts
COPY configs $WORKSPACE_DIR/configs

RUN pip install -r requirements.txt
RUN pip install torchaudio-0.13.0a0+597a735-cp37-cp37m-linux_x86_64.whl
RUN pip install --force-reinstall apex-0.1-cp37-cp37m-linux_x86_64.whl

# RUN git clone https://github.com/ROCmSoftwarePlatform/apex.git
# RUN cd apex && python setup.py install --cpp_ext --cuda_ext --transducer
