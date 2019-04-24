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

import model as m

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
    parser.add_argument('--image-file',
                        help='image for predication',
                        # required=True,
                        type=str)
    parser.add_argument('--save-transform-image',
                        help='Save temp image after transforms (True/False)',
                        action='store_true')
    parser.add_argument('--use-webcam',
                        help='Use webcam for predication',
                        action='store_true')
    parser.add_argument('--use-crop-mode',
                        help='Use crop mode for cropping person, that are you want to predict',
                        action='store_true')
    parser.add_argument('--gpus',
                        help='GPUs',
                        type=str)
    parser.add_argument('--min-confidence-threshold',
                        help='Minimum confidence threshold for drawing. Default: 0.5',
                        type=str)
    args = parser.parse_args()

    return args

refPt = []
cropping = False
tempPosition = ()

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
    global refPt, cropping, tempPosition
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
 
	# check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
        refPt.append((x, y))
        cropping = False
    else:
        tempPosition = (x, y)
   
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
    global refPt, tempPosition
    
    args = parse_args()
    model_file, num_layers, IMAGE_SIZE = loadConfig(args.cfg)
    
    transform_image = False
    use_webcam = False
    gpus = ''
    use_crop = False
    min_confidence_threshold = 0.5
    

    if args.image_file:
        image_file = args.image_file   
    if args.save_transform_image:
        transform_image = args.save_transform_image
    if args.use_webcam:
        use_webcam = args.use_webcam
    if args.gpus:
        gpus = args.gpus
    if args.use_crop_mode:
        use_crop = args.use_crop_mode
    if args.min_confidence_threshold:
        min_confidence_threshold = np.float(args.min_confidence_threshold)
        
    model = eval('m.get_pose_net')(
        num_layers, is_train=False
    )
    
    if model_file:
        print('=> loading model from {}'.format(model_file))
        model.load_state_dict(torch.load(model_file))
        if len(gpus) != 0:
            GPUS = [int(i) for i in gpus.split(',')]
            model = torch.nn.DataParallel(model, device_ids=GPUS).cuda()
    else:
        print('Error')
        return
        
    if use_webcam == False:
        ## Load an image
        data_numpy = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if data_numpy is None:
            raise ValueError('Fail to read image {}'.format(image_file))
        print(data_numpy.shape)
        
        if use_crop == True:
            cv2.namedWindow("image")
            cv2.setMouseCallback("image", click_and_crop)
            
            while True:
                key = cv2.waitKey(1) & 0xFF
             
                if len(refPt) == 2:
                    temp = data_numpy.copy()
                    cv2.rectangle(temp, refPt[0], refPt[1], (0, 255, 0), 2)
                    cv2.imshow("image", temp)
                    cv2.waitKey(1) & 0xFF
                    break
                elif len(refPt) == 1:
                    temp = data_numpy.copy()
                    cv2.rectangle(temp, refPt[0], tempPosition, (0, 255, 0), 2)
                    cv2.imshow("image", temp)
                else:
                    cv2.imshow("image", data_numpy)
                    
            data_numpy = data_numpy[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
            
        input = cv2.resize(data_numpy, (IMAGE_SIZE[0], IMAGE_SIZE[1]))

        # vis transformed image
        if transform_image == True:
            copyInput = input.copy()
            cv2.rectangle(copyInput, (np.int(IMAGE_SIZE[0]/2 + IMAGE_SIZE[0]/4), np.int(IMAGE_SIZE[1]/2 + IMAGE_SIZE[1]/4)), 
                                     (np.int(IMAGE_SIZE[0]/2 - IMAGE_SIZE[0]/4), np.int(IMAGE_SIZE[1]/2 - IMAGE_SIZE[1]/4)), (255,0,0), 2)
            cv2.imwrite('transformed.jpg', copyInput)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        input = transform(input).unsqueeze(0)
        
        # switch to evaluate mode
        model.eval()
        with torch.no_grad():
            # compute output heatmap
            output = model(input)
            coords, maxvals = m.get_max_preds(output.clone().cpu().numpy())
            print(maxvals)
            cv2.waitKey(1000) & 0xFF
            image = data_numpy.copy()
            for i in range(coords[0].shape[0]):
                mat = coords[0,i]
                x, y = int(mat[0]), int(mat[1])
                if maxvals[0, i] >= min_confidence_threshold:
                    cv2.circle(image, (np.int(x*data_numpy.shape[1]/output.shape[3]), 
                          np.int(y*data_numpy.shape[0]/output.shape[2])), 2, (0, 0, 255), 2)
                   
            cv2.imwrite('result.jpg', image)
            cv2.imshow('result.jpg', image)
            cv2.waitKey(2000) & 0xFF
        
        print('Success')
    else:
        sample = cv2.imread('sample.png', -1)
        alpha_s = sample[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        cap = cv2.VideoCapture(0)
        while(True):
            ret, data_numpy = cap.read()
            if not ret: break
                            
            input = cv2.resize(data_numpy, (IMAGE_SIZE[0], IMAGE_SIZE[1]))

            # vis transformed image
            if transform_image == True:
                copyInput = input.copy()
                cv2.rectangle(copyInput, (np.int(IMAGE_SIZE[0]/2 + IMAGE_SIZE[0]/4), np.int(IMAGE_SIZE[1]/2 + IMAGE_SIZE[1]/4)), 
                                         (np.int(IMAGE_SIZE[0]/2 - IMAGE_SIZE[0]/4), np.int(IMAGE_SIZE[1]/2 - IMAGE_SIZE[1]/4)), (255,0,0), 2)
                cv2.imwrite('transformed.jpg', copyInput)

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                ])
            input = transform(input).unsqueeze(0)
            
            # switch to evaluate mode
            model.eval()
            with torch.no_grad():
                # compute output heatmap
                output = model(input)
                coords, maxvals = m.get_max_preds(output.clone().cpu().numpy())
                image = data_numpy.copy()
                badPoints = 0
                for i in range(coords[0].shape[0]):
                    mat = coords[0,i]
                    x, y = int(mat[0]), int(mat[1])
                    if maxvals[0, i] >= min_confidence_threshold:
                        cv2.circle(image, (np.int(x*data_numpy.shape[1]/output.shape[3]), 
                              np.int(y*data_numpy.shape[0]/output.shape[2])), 2, (0, 0, 255), 2)
                    if maxvals[0, i] <= 0.4:
                        badPoints += 1
                if badPoints >= coords[0].shape[0]/3:
                    cv2.rectangle(image, (np.int(data_numpy.shape[1]/2 + data_numpy.shape[1]/4), np.int(data_numpy.shape[0]/2 + data_numpy.shape[0]/4)), 
                                         (np.int(data_numpy.shape[1]/2 - data_numpy.shape[1]/4), np.int(data_numpy.shape[0]/2 - data_numpy.shape[0]/4)), (255,0,0), 2)
                    for c in range(0, 3):
                        image[10:10+sample.shape[0], 10:10+sample.shape[1], c] = (alpha_s * sample[:, :, c] +
                                  alpha_l * image[10:10+sample.shape[0], 10:10+sample.shape[1], c])
                    cv2.putText(image, "locate your body as shown on images for keypoint detection", (10, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), lineType=cv2.LINE_AA)
                cv2.imshow('result', image)
            
            cv2.waitKey(10)
            #if cv2.waitKey(1) & 0xFF == ord('q'): break

        cv2.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()