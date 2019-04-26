# Demo of Simple Baselines for Human Pose Estimation and Tracking
![result.jpg](http://immage.biz/images/2019/04/26/SQ19.jpg)
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

# Export models for OpenVINO
For using OpenVINO script you shold convert original models into \*.onnx and then to \*.bin and \*.xml. For this you need:
1. Put export.py in pose_estimation folder of original repository
2. Run is like you run train.py or valid.py
3. In folder with model, that you put as argument in script, you can find new file \*.onnx
3. This file you shold insert, like argument for model optimizer of OpenVINO (OPENVINO_ROOT/deployment_tools/model_optimizer/mo.py). After this you get a \*.bin and \*.xml for openvino demo script

# Exmple of using
This repository includes an example image. example.png:
![example.png](http://immage.biz/images/2019/04/26/SQ1Y.png) <br>
If we try to run our script with a key --skip-crop-mode, we get an image without or wrong keypoints and message to console *"Bad position of person! Can't find key points. Please, place it at the center of the image or use crop mode for this"*. For correct prediction, we should place the person at the center of the image or use crop-mode for selecting him, if our image is not correct. For example, we can select our person like this:
![example.png](http://immage.biz/images/2019/04/26/SQ1o.png)<br>
And get the correct result:
![result.jpg](http://immage.biz/images/2019/04/26/SQ19.jpg)
