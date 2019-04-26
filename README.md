# Demo of Simple Baselines for Human Pose Estimation and Tracking

This repository contains testing code for the paper https://arxiv.org/abs/1804.06208 .  <br>
Original repository (https://github.com/Microsoft/human-pose-estimation.pytorch) <br>
This repository created at Intel R&D lab 

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
<code> python demo.py --cfg \<path to cfg *.yaml\> --image-file \<path to image\> [--save-transform-image] [--use-webcam] [--skip-crop-mode] [--gpus \<id\> ]  [--min-confidence-threshold \<coef\> ] </code>

Description of args:
* cfg (only for demo.py) : You should choose config with the same name as a model, that you are want to use. This file includes different configs for these models.
* model-file (only for openvino-demo.py) : You should set it to your *.xml model
* save-transform-image: You can set it for saving temp image after resizing and drawing bounding box (it works for webcam too)
* use-webcam : Use webcam for getting images for predict
* skip-crop-mode : Use crop mode for cropping person, that are you want (after adding this parameter, you get a new window with your photo, where you should highlight a required zone). By default crop mode is on. You can skip it with help this key
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
