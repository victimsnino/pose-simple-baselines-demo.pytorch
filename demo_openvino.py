import argparse
import os
import pprint
import sys

sys.path.append('C:\Program Files (x86)\IntelSWTools\openvino_2019.1.087\python\python3.6')

from openvino.inference_engine import IENetwork, IEPlugin
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
import numpy as np
import model as m

#model
IMAGE_SIZE = [256, 256]

image = np.empty(())


 # main and args
def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    
    parser.add_argument('--model-file',
                        help='Path to an .xml file with a trained mode',
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
    parser.add_argument('--gpu',
                        help='Use GPU',
                        action='store_true')
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
        
def main():
    global refPt, tempPosition
    args = parse_args()
    
    transform_image = False
    use_webcam = False
    gpu = False
    use_crop = False
    min_confidence_threshold = 0.5
    
    if args.model_file:
        model_xml = args.model_file
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
    if args.image_file:
        image_file = args.image_file   
    if args.save_transform_image:
        transform_image = args.save_transform_image
    if args.use_webcam:
        use_webcam = args.use_webcam
    if args.gpu:
        gpu = args.gpu
    if args.use_crop_mode:
        use_crop = args.use_crop_mode
    if args.min_confidence_threshold:
        min_confidence_threshold = np.float(args.min_confidence_threshold)
        

    if model_xml:
        print("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
        net = IENetwork(model=model_xml, weights=model_bin)
        net.batch_size = 1
    else:
        print('Error')
        return
    if gpu == True:
        plugin = IEPlugin('GPU')
    else:
        plugin = IEPlugin('CPU')
    
    # if plugin.device == "CPU":
        # supported_layers = plugin.get_supported_layers(net)
        # not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        # if len(not_supported_layers) != 0:
            # log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      # format(plugin.device, ', '.join(not_supported_layers)))
            # log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      # "or --cpu_extension command line argument")
            # sys.exit(1)
        # assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
        # assert len(net.outputs) == 1, "Sample supports only single output topologies"
    
    input_blob = next(iter(net.inputs))
    print(net.inputs['input_1'].shape)
    print("Loading model to the plugin")
    exec_net = plugin.load(network=net)
    print("Loaded")
    IMAGE_SIZE[0] = net.inputs['input_1'].shape[2]
    IMAGE_SIZE[1] = net.inputs['input_1'].shape[3]
    del net
    
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


        # compute output heatmap
        output = exec_net.infer(inputs={input_blob: input})['output1']
        coords, maxvals = m.get_max_preds(output)
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
            
            # compute output heatmap
            output = exec_net.infer(inputs={input_blob: input})['output1']
            coords, maxvals = m.get_max_preds(output)
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
            cv2.imshow('result', image)
        
            cv2.waitKey(10)
        #if cv2.waitKey(1) & 0xFF == ord('q'): break

        cv2.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()