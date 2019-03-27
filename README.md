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
<code> python demo.py --model-file \<path to model\> --image-file \<path to image\> --model-layers \<count of layers\> --model-input-size \<size of input layer\> [--save-transform-image] [--use-webcam] [--use-crop-mode] [--gpus \<id\> ]  [--min-confidence-threshold \<coef\> ] </code>

Description of args:
* model-layers: You should set this parameter relates to your model. For example, "pose_resnet_152_384x384.pth.tar" model has 152 layers 
* model-input-size: You should set this parameter relates to your model. For example, "pose_resnet_152_384x384.pth.tar" model has size 384
* save-transform-image: You can set it for saving temp image after resizing and drawing bounding box (it works for webcam too)
* use-webcam : Use webcam for getting images for predict
* use-crop-mode : Use crop mode for cropping person, that are you want (after adding this parameter, you get a new window with your photo, where you should highlight a required zone)
* min-confidence-threshold : Minumal confidence threshold of joints, that will be drawing on image. Default: 0.5

## Important! ##
Person for estimation must be at the center of image, else it can work wrong!<br>
Example:<br>
Bad positions:<br>
![Image of BadPosition](http://immage.biz/images/2019/03/07/SP53.jpg)
![Image of BadPosition](http://immage.biz/images/2019/03/07/SP5U.jpg)
<br> Good position: <br>
![Image of GoodPosition](http://immage.biz/images/2019/03/07/SP50.jpg)
![Image of GoodPosition](http://immage.biz/images/2019/03/07/SP5v.jpg)
## Note: ##
If you don't know, at the center of the image your person or not, you can use option --save-transform-image. After this, you get an image "transformed.jpg", where you can see a blue box. Your person must be into this box fully or most of the body  <br>
Examples: <br>
![Image of Good](http://immage.biz/images/2019/03/13/SPgD.jpg) ![Image of Good](http://immage.biz/images/2019/03/13/SPgF.jpg)
![Image of Bad](http://immage.biz/images/2019/03/13/SPgd.jpg) ![Image of Bad](http://immage.biz/images/2019/03/13/SPCH.jpg)
