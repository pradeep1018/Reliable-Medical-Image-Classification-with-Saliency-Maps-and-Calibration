from torchvision.models import densenet121
from torch_cka import CKA
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.nn.functional import one_hot
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from captum.attr import LayerGradCam, LayerAttribution, LRP
from tqdm import tqdm, trange
from torchvision.models import densenet121
from libauc.models import densenet121 as DenseNet121
from libauc.metrics import auc_roc_score
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import datetime
import pytz
from torch.utils.data import DataLoader, Subset

from losses import ClassDistinctivenessLoss, SpatialCoherenceConv, sigmoid_focal_loss
from libauc.losses import AUCM_MultiLabel
from libauc.optimizers import PESG
from metrics import AUC
from utils import EarlyStopping
from dataset import CheXpertData

model1 = densenet121(drop_rate=0)
model1.classifier = nn.Linear(1024, 5)
model1.load_state_dict(torch.load('model-weights/chexpert_wce.pt'))

model2 = densenet121(drop_rate=0)
model2.classifier = nn.Linear(1024, 5)
model2.load_state_dict(torch.load('model-weights/chexpert_dam.pt'))

train_csv = 'CheXpert-v1.0-small/train.csv'
valid_csv = 'CheXpert-v1.0-small/valid.csv'
train_data = CheXpertData(train_csv, mode='train')
val_data = CheXpertData(valid_csv, mode='train') #changed to train mode

img_ind = 244
qimg = train_data[img_ind][0]
qimg_label = train_data[img_ind][1]



cka = CKA(model1, model2,
          model1_name="wce",   # good idea to provide names to avoid confusion
          model2_name="dam",   
          model1_layers='denseblock4.denselayer16.conv2', # List of layers to extract features from
          model2_layers='denseblock4.denselayer16.conv2', # extracts all layer features by default
          device='cuda')

cka.compare(DataLoader(val_data,
                            drop_last=True,shuffle=False,
                            batch_size=16, num_workers=32, pin_memory=True), DataLoader(val_data,
                            drop_last=True,shuffle=False,
                            batch_size=16, num_workers=32, pin_memory=True)) # secondary dataloader is optional

results = cka.export()  # returns a dict that 