import argparse
import os
import pprint

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
import numpy as np


import yaml
from easydict import EasyDict as edict

import model as mod

#model
IMAGE_SIZE = [256, 256]

image = np.empty(())

 # main and args
def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    
    parser.add_argument('--cfg',
                        help='Path to cfg *.yaml',
                        required=True,
                        type=str)
    args = parser.parse_args()

    return args

def loadConfig(config_file):
    numLayers = 0 
    size = ()
    model_file = ''
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k == 'PRETRAINED':
                model_file = v
            if k == 'NUM_LAYERS':
                numLayers= v
            if k == 'IMAGE_SIZE':
                size = (v[0], v[1])
    return model_file, numLayers, size
  

def main():  
    args = parse_args()
    model_file, num_layers, IMAGE_SIZE = loadConfig(args.cfg)
    
    model = eval('mod.get_pose_net')(
        num_layers, is_train=False
    )

    if model_file:
        print('=> loading model from {}'.format(model_file))
        model.load_state_dict(torch.load(model_file))
    
    dummy_input = torch.randn(1, 3,IMAGE_SIZE[0], IMAGE_SIZE[1])
    
    input_names = [ "input_1" ]
    output_names = [ "output1" ]

    torch.onnx.export(model, dummy_input, model_file+".onnx", verbose=True, input_names=input_names, output_names=output_names)

    print('END')


if __name__ == '__main__':
    main()
