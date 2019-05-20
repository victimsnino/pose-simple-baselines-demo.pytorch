# Demo of Simple Baselines for Human Pose Estimation and Tracking
This repository contains demo code for the paper [Simple Baselines for Human Pose Estimation and Tracking](https://arxiv.org/abs/1804.06208). <br>
Original repository (https://github.com/Microsoft/human-pose-estimation.pytorch) <br>
This repository created at Intel R&D lab at HSE, Nizhny Novgorod.
 <br> ![result.jpg](http://immage.biz/images/2019/04/26/SQ19.jpg) <br>

# Introduction
This code estimates keypoints of one person, which centered on the image (inside blue rectangle)

# Requirements
* Python 3.6
* PyTorch 0.4.1 (should also work with 1.0, but not tested)
* (Optional) Install dependencies from original repository

# Testing
1. Download required models from original repository from step 8 of Installation (for example [256x256 model](https://drive.google.com/open?id=1V2AaVpDSn-eS7jrFScHLJ-wvTFuQ0-Dc) 
2. Prepare image, that you want to use for testing
3. run script:<br>
<code> python demo.py --cfg \<path to cfg *.yaml\> --image-file \<path to image\> [--save-transform-image] [--skip-crop-mode] [--gpus \<id\> ]  [--min-confidence-threshold \<coef\> ] </code>

Description of args:
* cfg (only for demo.py) : You should choose config with the same name as a model, that you are want to use. This file includes different configs for these models.
* image-file : Path to your image for predication. You can didn't use this argument, then will be used webcam
* model-file (only for openvino-demo.py) : You should set it to your *.xml model
* save-transform-image: You can set it for saving temp image after resizing and drawing bounding box (it works for webcam too)
* skip-crop-mode : Use crop mode for cropping person, that are you want (after adding this parameter, you get a new window with your photo, where you should highlight a required zone). By default crop mode is on. You can skip it with help this key
* min-confidence-threshold : Minumal confidence threshold of joints, that will be drawing on image. Default: 0.5

# Export models for OpenVINO
For using OpenVINO script you shold convert original models into \*.onnx and then to \*.bin and \*.xml. For this you need:
1. Put models like for using in ./models
2. Run export.py with --cfg argument
3. In folder with models you can find new file \*.onnx
3. This file you shold insert, like argument for model optimizer of OpenVINO (OPENVINO_ROOT/deployment_tools/model_optimizer/mo.py). After this you get a \*.bin and \*.xml for openvino demo script

# Example of using
This repository includes an example image. example.png: <br>
![example.png](http://immage.biz/images/2019/04/26/SQ1Y.png) <br>
If we try to run our script with a key --skip-crop-mode, we get an image without or wrong keypoints and message to console *"Bad position of person! Can't find key points. Please, place it at the center of the image or use crop mode for this"*. For correct prediction, we should place the person at the center of the image or use crop-mode for selecting him, if our image is not correct. For example, we can select our person like this: <br>
![example.png](http://immage.biz/images/2019/04/26/SQ1o.png)<br>
And get the correct result: <br>
![result.jpg](http://immage.biz/images/2019/04/26/SQ19.jpg)
