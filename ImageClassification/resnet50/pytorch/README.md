# ResNet50 v1.5

## The model
The ResNet50 v1.5 model is a modified version of the [original ResNet50 v1 model](https://arxiv.org/abs/1512.03385).

The difference between v1 and v1.5 is that, in the bottleneck blocks which requires
downsampling, v1 has stride = 2 in the first 1x1 convolution, whereas v1.5 has stride = 2 in the 3x3 convolution.

This difference makes ResNet50 v1.5 slightly more accurate (~0.5% top1) than v1, but comes with a smallperformance drawback (~5% imgs/sec).

The model is initialized as described in [Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification](https://arxiv.org/pdf/1502.01852.pdf)

## Training procedure

### Optimizer

This model trains for 39 epochs, with standard ResNet v1.5 setup:

* LARS with momentum (0.9)

* Learning rate = 9.1 for 256 batch size, for other batch sizes we lineary
scale the learning rate.

* Learning rate schedule - we use polynomial LR schedule

* For bigger batch sizes (512 and up) we use linear warmup of the learning rate
during first couple of epochs
according to [Training ImageNet in 1 hour](https://arxiv.org/abs/1706.02677).
Warmup length depends on total training length.

* Weight decay: 0.0002

* We do not apply WD on Batch Norm trainable parameters (gamma/bias)

* Label Smoothing: 0.1

* We train for:

    * 39 Epochs -> configuration that reaches 75.9% top1 accuracy

### Data Augmentation

This model uses the following data augmentation:

* For training:
  * Normalization
  * Random resized crop to 224x224
    * Scale from 8% to 100%
    * Aspect ratio from 3/4 to 4/3
  * Random horizontal flip

* For inference:
  * Normalization
  * Scale to 256x256
  * Center crop to 224x224

# Quick start guide

## Geting the data

The ResNet50 v1.5 script operates on ImageNet 1k, a widely popular image classification dataset from ILSVRC challenge.

PyTorch can work directly on JPEGs, therefore, preprocessing/augmentation is not needed.

1. Download the images from http://image-net.org/download-images

2. Extract the training data:
  ```bash
  mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
  tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
  find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
  cd ..
  ```

3. Extract the validation data and move the images to subfolders:
  ```bash
  mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
  wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
  ```

The directory in which the `train/` and `val/` directories are placed, is referred to as `<path to imagenet>` in this document.

## Running training

mkdir $HOME/dockerx
#change directory
cd $HOME/dockerx
 
# clone v2.0 repository
git clone -b mlperf-v2.0 https://github.com/ROCmSoftwarePlatform/MLPerf-mGPU.git
 
# Enter the ROCm Pytorch docker container to run the code
alias ptdrun='sudo docker run -it --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $HOME/dockerx:/dockerx -w /dockerx --shm-size=64G'
 
ptdrun <rocm/pytorch container>
 
# Install dependencies to run the ResNet50 code
cd MLPerf-mGPU/image_classification
./requirements.sh
 
cd MLPerf-mGPU/utils
python3.7 setup.py install

#Change directory
 
cd MLPerf-mGPU/image_classification

Ensure imagenet is mounted in the `/data/imagenet` directory.

To run the workload:

* FP16 NHWC:
      `./RN50_AMP_LARS_8GPUS_NHWC.sh`

* FP16 NCHW:
      `./RN50_AMP_LARS_8GPUS_NCHW.sh`

* FP32 NCHW:
     `./RN50_FP32_8GPUS_NCHW.sh`

To run the script on lesser GPUs, change the value of '--nproc_per_node' in the above 8 GPUs script.
Use `python main.py -h` to obtain the list of available options in the `main.py` script.
