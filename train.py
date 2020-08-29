# Imports here

"""
    EXAMPLE: RUN python train.py flowers --save_directory Save_dir --architecture densenet121 --epochs 5 --learnrate 0.003 --gpu cuda 
"""

import argparse 
parser= argparse.ArgumentParser(description='Train a model')
parser.add_argument('data_directory',type=str,help='specify data folder')
parser.add_argument('--save_directory',type=str,help='where to save checkpoints')
parser.add_argument('--architecture',type=str,help='set the pretrained model to use')

parser.add_argument('--epochs',type=int, help='set the number of epochs for training')
parser.add_argument('--learnrate',type=float,help='set the learning rate')
parser.add_argument('--gpu',type=str,help='use gpu')

args=parser.parse_args()

#args.name to use it in code

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

data_dir=args.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(244),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

test_val_transforms= transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_data =datasets.ImageFolder(train_dir,train_transforms)
test_data=datasets.ImageFolder(test_dir,test_val_transforms)
val_data=datasets.ImageFolder(valid_dir,test_val_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data,batch_size=64, shuffle=True)
testloader=torch.utils.data.DataLoader(test_data,batch_size=64)
validationloader=torch.utils.data.DataLoader(val_data,batch_size=64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #to use GPU

arch= args.architecture
if arch=='densenet121':  
    model = models.densenet121(pretrained=True)
elif arch=='vgg13':
    model = models.vgg13(pretrained=True)
else:
    print('I do not support this model yet')

print("pretrained model downloaded")
for param in model.parameters():
    param.requires_grad = False


model.classifier = nn.Sequential(nn.Linear(1024, 500),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(500, 200),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(200, 102),                     #dataset has 102 categories.
                                     nn.LogSoftmax(dim=1))



criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learnrate)    #acting only on model.classifier parameters!!

model.to(args.gpu) 

epochs= args.epochs 

train_losses, val_losses = [], []



step_two=0
for e in tqdm(range(epochs)):
    step_two+=1
    #training step:
    step=0
    running_loss=0
    print("Training in progress. Epoch {}/{}:".format(step_two,epochs))


    for images, labels in trainloader:                             #TRAINING STEP BEGINS

        step+=1

        images, labels = images.to(args.gpu), labels.to(args.gpu)       #move tensors to the GPU

        optimizer.zero_grad()                                       #it is always necessary to reset the error 

        log_ps= model.forward(images)                         
        loss = criterion(log_ps, labels)                            #we apply the criterion previously set to predictions vs actual 
        loss.backward()                                             #backpropagation
        optimizer.step()                                            #updates weights !!! IMPORTANT 
        
        running_loss+=loss.item()







        #evaluation step:                                                       
    else: 
        model.eval()                                                #disables dropouts
        val_loss=0
        accuracy=0

        with torch.no_grad():                                 
            for images, labels in validationloader:                 #EVALUATION STEP BEGINS




                images, labels = images.to(args.gpu), labels.to(args.gpu)

                val_log_ps=model(images)
                val_loss+=criterion(val_log_ps,labels)

                #accuracy calculation:
                prob=torch.exp(val_log_ps)                          #Get probabilities from log(prob) returned by softmax
                top_p, top_class =prob.topk(1, dim=1)               #Get the top class and corresponding probability for each image

                equality= top_class==labels.view(*top_class.shape)  #returns 1 if the prediction matches labels, else 0. 
                                                                        #accuracy is just the mean of all this values


                accuracy+= torch.mean(equality.type(torch.FloatTensor))

            model.train()                                           #switches back to train mode, enables dropouts.




        val_losses.append(val_loss/len(testloader))
        train_losses.append(running_loss/len(validationloader))



        print("Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
              "Validation Loss: {:.3f}.. ".format(val_loss/len(validationloader)),
              "Validation Accuracy: {:.3f}".format(accuracy/len(validationloader)))
test_accuracy=0
model.eval()  
with torch.no_grad():                                 
    for images, labels in testloader:                

        images, labels = images.to(device), labels.to(device)

        test_log_ps=model(images)


        test_prob=torch.exp(test_log_ps)                         
        top_p, top_class =test_prob.topk(1, dim=1)               



        test_equality= top_class==labels.view(*top_class.shape)  



        test_accuracy+= torch.mean(equality.type(torch.FloatTensor))

print("Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
checkpoint= {'pre trained model': args.architecture,

            'epochs':5,
            'accuracy_test':0.825,
            'input classifier': 1024,
            'hidden layers':[500,200], 
            'output classifier': 102,
            'parameters_dict': model.state_dict(),
            'optimizer_state' : optimizer.state_dict,
            'class_to_idx':train_data.class_to_idx,    
    }

import os 
os.chdir( args.save_directory) 
torch.save(checkpoint,'checkpoint.pht' )


print("Model checkpoint saved at: {}".format(args.save_directory))