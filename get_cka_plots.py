import os
import matplotlib.pyplot as plt
import numpy as np

import torch
from torchvision.models import densenet121
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm

from dataset import CheXpertData
from torch.utils.data import DataLoader, Subset
from captum.attr import Saliency
from captum.attr import LayerGradCam, LayerAttribution, LRP
from captum.attr import visualization as viz

#loading data
train_csv = 'CheXpert-v1.0-small/train.csv'
valid_csv = 'CheXpert-v1.0-small/valid.csv'
train_data = CheXpertData(train_csv, mode='train')
val_data = CheXpertData(valid_csv, mode='train')

img_ind = 244
qimg = train_data[img_ind][0]
qimg_label = train_data[img_ind][1]
print(qimg_label)

# qpath = f'/cka_plot_results/img{img_ind}'
# my_path = os.path.abspath(__file__) 
# my_path = my_path[0:-20]
# my_path += qpath 
# if not os.path.exists(my_path):
#     os.mkdir(my_path)

model_type = 'wce'
model = densenet121()
model.classifier = nn.Linear(1024, 5) #need to generalize for number of classes        
model.load_state_dict(torch.load(f'model-weights/chexpert_{model_type}.pt'))

last_cnn_layer = model.features.denseblock4.denselayer16.conv2
representations = []

def hook(module, input, output):
    representations.append(output)

handle = last_cnn_layer.register_forward_hook(hook)

model = model.eval()
model_input = qimg.unsqueeze(0)
model_output = model(model_input)

representations = representations[0].detach()

print(representations.shape)

handle.remove()