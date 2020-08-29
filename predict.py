

"""
    EXAMPLE: RUN python predict.py flowers/test/15 image_06369 Save_dir --top_k 1  --gpu cuda 
"""

import PIL as PIL
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import time



import argparse 
parser= argparse.ArgumentParser(description='Predict a flower')
parser.add_argument('img_directory',type=str,help='specify image location')
parser.add_argument('img_file',type=str,help='specify image file name')
parser.add_argument('load_checkpoint',type=str,help='where to load checkpoint from')


parser.add_argument('--top_k',type=int, help='number of classes with highest probabilities')

parser.add_argument('--gpu',type=str,help='use gpu')

args=parser.parse_args()


def load_checkpoint(filepath):
    
   
    
    checkpoint=torch.load(filepath)
    
    
    model=models.densenet121(pretrained=True)
    model.classifier= nn.Sequential(nn.Linear(1024, 500),
                          nn.ReLU(),
                          nn.Dropout(0.2),
                          nn.Linear(500, 200),
                          nn.ReLU(),
                          nn.Dropout(0.2),
                          nn.Linear(200, 102),                     
                          nn.LogSoftmax(dim=1))
    
    
    model.load_state_dict(checkpoint['parameters_dict'])
    
    return model



import os
os.chdir(args.load_checkpoint) 
checkpoint=torch.load('checkpoint.pht')

val_dict=checkpoint['class_to_idx']
def process_image(imagepath):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image=Image.open(imagepath)
    ratio= np.max(image.size)/np.min(image.size)

    if np.argmin(image.size)==0:
        box=[256, int(ratio*256)]
    else:
        box=[int(ratio*256),256]

    image=image.resize(box)
    image=image.crop(box=(image.width/2-112,image.height/2-112,image.width/2+112,image.height/2+112))

    np_image = np.array(image)
    np_image = np_image/255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image=  (np_image - mean)/std
    
    
    transpose_image=np_image.transpose((2, 0, 1))
    tensor=torch.from_numpy(transpose_image)
    
    
    return  tensor

def predict(image_path, model,):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to('cpu')
    
    tensor_img=process_image(image_path)
    tensor_img=tensor_img.unsqueeze_(dim=0)
    tensor_img=tensor_img.float()
    
    tensor_img
    prediction=model.forward(tensor_img)
    
    prob=torch.exp(prediction)
    top_p, top_class= prob.topk(args.top_k)
    
    return top_class, top_p
    # TODO: Implement the code to predict the class from an image file
import json

os.chdir('/home/workspace/ImageClassifier')
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def return_type(top_class):
    keys=[]
    flower_type=[]
    for values in top_class[0][0]:
        for item in val_dict.items():
            if item[1] == values:
               
                flower_type.append(cat_to_name[item[0]])
                keys.append(item[0])

    print(keys,flower_type )

os.chdir('/home/workspace/ImageClassifier/' + args.load_checkpoint)
    
loaded_model= load_checkpoint('checkpoint.pht')
checkpoint=torch.load('checkpoint.pht')

os.chdir('/home/workspace/ImageClassifier/'+args.img_directory)

top_class= predict(args.img_file+'.jpg', loaded_model)
return_type(top_class)