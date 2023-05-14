import os
# import shutil
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
val_data = CheXpertData(valid_csv, mode='train') #changed to train mode

img_ind = 600
qimg = train_data[img_ind][0]
qimg_label = train_data[img_ind][1]
print(qimg_label)
# plt.imshow((qimg.permute(1, 2, 0).to(torch.double) * 255).to(torch.uint8)) #IMAGES ARE WILDLY BLACK IN FLOAT
# plt.savefig('temp.png')

qpath = f'/saliency_results/img{img_ind}'
my_path = os.path.abspath(__file__) 
my_path = my_path[0:-20]
my_path += qpath 
# print(my_path)   
if not os.path.exists(my_path):
    os.mkdir(my_path)


# types = ['wce', 'wce_cd', 'double_stage']
types = ['wce', 'vgg19', 'resnet101']
for model_type in types:    
    print(f'Running for Model:{model_type}')
    if model_type == 'resnet101':
        model = models.resnet101(weights = 'DEFAULT')
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(2048, 5)
    elif model_type == 'vgg19':
        model = models.vgg19(weights = 'DEFAULT')
        model.classifier = nn.Linear(25088, 5)                                    
    else:
        model = densenet121()
        model.classifier = nn.Linear(1024, 5) #need to generalize for number of classes        
    model.load_state_dict(torch.load(f'model-weights/chexpert_{model_type}.pt'))
    model = model.eval()
    
    #Somehow get model's predictions    
    sigmoid = nn.Sigmoid()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model(qimg.unsqueeze(0))
    predictions = sigmoid(outputs).to(device)
    print(predictions)

    #Generate Saliency Maps corresponding to each label
    if model_type == 'resnet101':
        gradcam = LayerGradCam(model, layer=model.layer4[2].conv3)
    elif model_type == 'vgg19':
        gradcam = LayerGradCam(model, layer=model.features[34])
    else:
        gradcam = LayerGradCam(model, layer=model.features.denseblock4.denselayer16.conv2)    
    # lrp = LRP(model)
    attr_classes = [torch.Tensor(gradcam.attribute(inputs=qimg.unsqueeze(0), target = [i])).to(device) for i in range(5)]
    for disease_class in range(5):
        act_map = viz.visualize_image_attr(attr_classes[disease_class][0].cpu().permute(1, 2, 0).detach().numpy())
        # plt.savefig('temp2.png')
        upsampled_map = LayerAttribution.interpolate(attr_classes[disease_class], qimg.unsqueeze(0).shape[2:], interpolate_mode = 'bilinear')
        # upsampled_map = attr_classes
        print(attr_classes[disease_class].shape)
        print(upsampled_map.shape)
        print(qimg.unsqueeze(0).shape)
        _ = viz.visualize_image_attr_multiple(upsampled_map[0].cpu().permute(1,2,0).detach().numpy(),
                                            qimg.permute(1,2,0).numpy(),
                                            ["original_image","blended_heat_map","masked_image"],
                                            ["all","positive","positive"],
                                            show_colorbar=True,
                                            titles=["Original", "Positive Attribution", "Masked"],
                                            fig_size=(18, 6))    
        plt.savefig(f'{my_path}/{model_type}_img{img_ind}_class_{disease_class}.png')
        plt.close()