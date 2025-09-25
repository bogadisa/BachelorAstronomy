"""
The majority of this code has been taken from the solution to exercise 5. I have then modified and implimented new features.
"""
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
#import matplotlib.pyplot as plt
import time
import os

#import skimage.io
import PIL.Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from functions import *
from tests import test_disjointed

seed = 1234
torch.manual_seed(seed)

class environmentClass(Dataset):
    def __init__(self, root_dir, output_dir, trvaltest, transform=None):
        self.root_dir = root_dir

        self.transform = transform
        self.imgfilenames=[]
        self.labels=[]
        self.classification = [name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))]


        if trvaltest==0:
        #load training data
            data_type = "trainfile.txt"
        elif trvaltest==1:
        #load validation data
            data_type = "testfile.txt"
        elif trvaltest==2:
        #load test data
            data_type = "valfile.txt"
        else:
            raise Exception(f"Expected trvaltest value between [0, 2], got {trvaltest}")

        path = os.path.join(output_dir, data_type)
        with open(path) as f:
            lines = f.readlines()
            for line in lines:
                filename, label = line.strip().split(" ")
                self.imgfilenames.append(os.path.join(root_dir, label, filename))
                # self.labels.append(self.onehot(label, self.classification))
                self.labels.append(self.classification.index(label))
        

    def __len__(self):
        return len(self.imgfilenames)

    def __getitem__(self, idx):
        image = PIL.Image.open(self.imgfilenames[idx]).convert('RGB')

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        label = torch.tensor(self.labels[idx])

        sample = {'image': image, 'label': label, 'filename': self.imgfilenames[idx]}
        return sample

    @staticmethod
    def onehot(label, classifications):
        return [int(c == label) for c in classifications]



def runstuff_finetunealllayers():
    #someparameters
    batchsize_tr = 16
    batchsize_test = 16
    maxnumepochs = 5
    numcl = 6

    # device=torch.device('cuda')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # quit()

    data_transforms = {}
    data_transforms['train']=transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    data_transforms['val']=transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    datasets = {}
    datasets['train'] = environmentClass(root_dir=img_dir, output_dir=output_dir, trvaltest=0, transform=data_transforms['train'])  
    datasets['val'] = environmentClass(root_dir=img_dir, output_dir=output_dir, trvaltest=1, transform=data_transforms['val'])
    datasets['test'] = environmentClass(root_dir=img_dir, output_dir=output_dir, trvaltest=2, transform=data_transforms['val'])


    dataloaders = {}
    dataloaders['train']= DataLoader(datasets['train'], batch_size=batchsize_tr, shuffle=True) 
    dataloaders['val']= DataLoader(datasets['val'], batch_size=batchsize_test, shuffle=False )
    dataloaders['test']= DataLoader(datasets['test'], batch_size=batchsize_test, shuffle=False)  

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, numcl)
    model.fc.reset_parameters()

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')

    lrates=[0.1, 0.01, 0.001]

    best_hyperparameter= None
    weights_chosen = None
    bestmeasure = None
    best_fig = None
    for lr in lrates:

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        best_epoch, best_perfmeasure, bestweights, fig = train_model_nocv_sizes(dataloader_train = dataloaders['train'], dataloader_test = dataloaders['val'] ,  model = model ,  
                                                                           losscriterion = criterion , optimizer = optimizer, scheduler = None, num_epochs = maxnumepochs , 
                                                                           device = device, classifications=classification)

        if best_hyperparameter is None:
            best_hyperparameter = lr
            weights_chosen = bestweights
            bestmeasure = best_perfmeasure
            best_fig = fig
        elif best_perfmeasure > bestmeasure:
            best_hyperparameter = lr
            weights_chosen = bestweights
            bestmeasure = best_perfmeasure
            best_fig = fig

    model.load_state_dict(weights_chosen)

    #saves a plot of the training and validation loss of the best chosen model
    fig_name = f"model_lr{best_hyperparameter:.0E}.png"
    best_fig.suptitle(rf"Model with learning rate $\lambda={best_hyperparameter}$")
    best_fig.savefig(os.path.join(output_dir, fig_name))


    accuracy, testloss = evaluate_acc(model = model , dataloader  = dataloaders['test'], losscriterion = criterion, device = device, classifications=classification, testing=True)
    #saves the best model
    torch.save(model, os.path.join(output_dir, "final_model.pt"))

    print('accuracy val', bestmeasure, 'accuracy test', accuracy)

def runstuff():
    batchsize_test = 16

    criterion = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(os.path.join(output_dir, "final_model.pt"), map_location=device)
    
    data_transforms = {}
    data_transforms['val']=transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    datasets = {}
    datasets['test'] = environmentClass(root_dir=img_dir, output_dir=output_dir, trvaltest=2, transform=data_transforms['val'])
    dataloaders = {}
    dataloaders['test']= DataLoader(datasets['test'], batch_size=batchsize_test, shuffle=False)  
    

    accuracy, testloss = evaluate_acc(model = model , dataloader  = dataloaders['test'], losscriterion = criterion, device = device, classifications=classification, testing=True)
    
    print('accuracy test', accuracy)

root_dir = os.path.normpath("/".join(__file__.split("\\")[:-1]))
#Whenever I run the code on my computer, I dont have a GPU
#So if there is a GPU, it means I can assume we are running the file on one of the nodes
if torch.cuda.is_available():
    img_dir = "/itf-fi-ml/shared/courses/IN3310/mandatory1_data"
else:
    img_dir = os.path.join(root_dir, "mandatory1_data")
    
if __name__ == "__main__":
    #generates a txt file with all image names and their classification
    #format: img_nr.jpg, classification

    data_filename = "all.txt"
    output_dir = os.path.join(root_dir, "output_folder")
    data_path = os.path.join(output_dir, data_filename)

    if not(os.path.isdir(output_dir)):
        os.mkdir(output_dir)

    classification = classify(img_dir, output_dir)

    data = np.loadtxt(data_path, dtype=str)

    X, X_test, y, y_test = train_test_split(data[:, 0], data[:, 1], random_state=seed, test_size=int(3e3))
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=seed, test_size=int(2e3))

    test_disjointed([X_train, X_test, X_val])

    #generate a similiar file to all.txt, but each file has a special use
    #format: img_nr.jpg, classification

    # if not(os.path.isfile(os.path.join(img_dir, "train.txt"))):
    split_txt(img_dir, output_dir, "trainfile.txt", X_train, y_train)
    # if not(os.path.isfile(os.path.join(img_dir, "testfile.txt"))):
    split_txt(img_dir, output_dir, "testfile.txt", X_test, y_test)
    # if not(os.path.isfile(os.path.join(img_dir, "valfile.txt"))):
    split_txt(img_dir, output_dir, "valfile.txt", X_val, y_val)

    # runstuff_finetunealllayers()
    runstuff()