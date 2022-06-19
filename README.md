# Deep-Neural-Network

In this coding assignment, we need to implement the deep neural network by any deep learning framework, e.g., Pytorch, TensorFlow, or Keras, then train the DNN model by the Cifar-10 dataset and try to beat the baseline performance.

[Sample code and Dataset](https://github.com/NCTU-VRDL/CS_AT0828/tree/main/HW5)

## Introduction

This repository use [CaiT](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/cait.py) model of [PyTorch Image Models (timm)](https://github.com/rwightman/pytorch-image-models) and SGD optimizer.


## Environment

Ubuntu 18.04.5 LTS
Intel® Core™ i7-3770 CPU @ 3.40GHz × 8
GeForce GTX 1080/PCIe/SSE2

## Requirement

You can just use this command: 
```env
conda env create -f environment.yml
```

or the following commands:

```env
conda create -n PRenv
conda activate PRenv
conda install pytorch=1.10.0 torchvision=0.11.1 -c pytorch
conda install matplotlib
conda install tqdm
pip install sklearn
pip install timm
```

## Repository Structure

The repository structure is:
```
Deep-Neural-Network(root)
  +-- x_test.npy          # testing data
  +-- x_train.npy         # training data
  +-- y_test.npy          # testing label
  +-- x_test.npy          # training label
  +-- HW5.pdf
  +-- HW5.py
  +-- best.pt             # model weight
  +-- environment.yml
```

## Training

To train the model, you need modify the 'TRAIN' parameter to True of HW5.py, and then run this command:

```train
python HW5.py
```

## Testing

Need your best.pt (model weight) and the four npy files of the dataset, and just run this command:

```train
python HW5.py
```

## Results

Our model achieves 0.9734

## Reference
[1] [PyTorch Image Models (timm)](https://github.com/rwightman/pytorch-image-models)
[2] [PyTorch Tutorial](https://github.com/pytorch/tutorials)
