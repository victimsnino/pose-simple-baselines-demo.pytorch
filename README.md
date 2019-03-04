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
2. python demo.py --input_model 

