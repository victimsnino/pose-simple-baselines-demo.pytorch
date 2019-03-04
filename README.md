# Simple Baselines for Human Pose Estimation and Tracking (sample)

This repository contains testing code for the paper https://arxiv.org/abs/1804.06208 .  <br>
Original repository (https://github.com/Microsoft/human-pose-estimation.pytorch)

# Introduction
This demo created for quick-testing original models for mpii dataset (other datasets and models not tested) by your own images. My code draws joints, which founded by model, on your image and save it as another image. This code doesn't use any detection models, therefore searches joints by a center person of your image

# Requirements
* Python 3.6
* PyTorch 0.4.1 (should also work with 1.0, but not tested)
* (Optional) Install dependencies from original repository

# Testing
1. Download required models from original repository from step 8 of Installation
2. Prepare image, that you want to use for testing
3. run script:<br>
<code> python demo.py --model-file <path to model> --image-file <path to image> --model-layers <count of layers> --model-input-size <size of input layer> [--save-transform-image]</code>
<br>
## Description of args:
  * --model-layers   - You should set this parameter relates to your model. For example, "pose_resnet_152_384x384.pth.tar" model has 152 layers 
  * --model-input-size - You should set this parameter relates to your model. For example, "pose_resnet_152_384x384.pth.tar" model has size 384
  * --save-transform-image  - You can set it for saving temp image after resizing

