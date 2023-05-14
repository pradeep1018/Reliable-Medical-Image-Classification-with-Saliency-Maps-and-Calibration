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
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import datetime
import pytz
from libauc.losses import AUCM_MultiLabel
from libauc.optimizers import PESG
from libauc.models import densenet121 as DenseNet121
from libauc.metrics import auc_roc_score
from torchmetrics.classification import BinaryCalibrationError

from losses import ClassDistinctivenessLoss, SpatialCoherenceConv, sigmoid_focal_loss
from metrics import AUC
from utils import EarlyStopping

def decf(epoch):
    if epoch == 1:
        return 10
    elif epoch <= 10:
        if epoch%2 == 0:
            return 1
        else:
            return 5
    return 2

class Trainer:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_classes = args.n_classes
        self.batch_size = args.batch_size
        self.data = args.data
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.drop_rate = args.drop_rate
        self.training_type = args.training_type
        self.wceloss = args.wceloss
        self.focalloss = args.focalloss
        self.gamma = args.gamma
        self.alpha = args.alpha
        self.cdloss = args.cdloss
        self.cdloss_weight = args.cdloss_weight
        self.scloss = args.scloss
        self.scloss_weight = args.scloss_weight
        self.grid_search = args.grid_search

        if self.training_type == 'double-stage':
            self.model = densenet121(drop_rate=self.drop_rate)
            self.model.classifier = nn.Linear(1024, self.n_classes)
            self.model.load_state_dict(torch.load('model-weights/{}_wce.pt'.format(self.data)))
        if self.training_type == 'dam':# Deep AUC Maximization
            # self.model = densenet121(drop_rate=self.drop_rate)
            # self.model = DenseNet121(pretrained=True, last_activation=None, activations='relu', num_classes=5)
            self.model = densenet121(drop_rate=self.drop_rate)
            self.model.classifier = nn.Linear(1024, self.n_classes)
            self.model.load_state_dict(torch.load('model-weights/{}_wce.pt'.format(self.data)))            
            # self.model = densenet121(weights='DEFAULT', drop_rate=self.drop_rate)
            # self.model.classifier = nn.Linear(1024, self.n_classes)
            #   
        elif self.training_type == 'vgg19':
            # self.model = models.__dict__['vgg'](num_classes = 5)
            self.model = models.vgg19(weights = 'DEFAULT')
            # pretrained_state = model_zoo.load_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth')
            self.model.classifier = nn.Linear(25088, self.n_classes)
        elif self.training_type == 'resnet101':
            self.model = models.resnet101(weights = 'DEFAULT')
            self.model.avgpool = nn.AdaptiveAvgPool2d(1)
            self.model.fc = nn.Linear(2048, self.n_classes)
            # self.model.classifier = nn.Linear(2048, self.n_classes)
        else:
            self.model = densenet121(weights='DEFAULT', drop_rate=self.drop_rate)
            self.model.classifier = nn.Linear(1024, self.n_classes)
        self.model = self.model.to(self.device)
        if self.training_type == 'dam':
            self.optimizer = None
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.sigmoid = nn.Sigmoid()
        if 1:
            self.metrics = AUC()
        else:
            self.metrics = auc_roc_score
        self.early_stopping = EarlyStopping(args)
        self.ece = BinaryCalibrationError()

        self.train_losses = []
        self.val_losses = []
        self.train_aucs = []
        self.val_aucs = []
        self.ece_vals = []

        self.loss_df = pd.DataFrame(columns=['Training Loss','Validation Loss'])
        if args.data=='chexpert':
            self.train_auc_df = pd.DataFrame(columns=['Atelectasis','Cardiomegaly','Consolidation','Edema','Plural Effusion','Mean'])
            self.val_auc_df = pd.DataFrame(columns=['Atelectasis','Cardiomegaly','Consolidation','Edema','Plural Effusion','Mean'])
        if args.data=='breastmnist':
            self.train_auc_df = pd.DataFrame(columns=['class1','class2','Mean'])
            self.val_auc_df = pd.DataFrame(columns=['class1','class2','Mean'])
        self.loss_batch_df = pd.DataFrame(columns=['custom loss','cross entropy','class distinctiveness','spatial coherence'])
        self.ece_val_df = pd.DataFrame(columns=['Atelectasis','Cardiomegaly','Consolidation','Edema','Plural Effusion','Mean'])

    def train_one_epoch(self):
        self.model.train()        
        losses = []
        y_true = torch.tensor([]).to(self.device)
        y_pred = torch.tensor([]).to(self.device)  
        for inputs, targets in tqdm(self.train_loader):            
            self.optimizer.zero_grad()                    
            inputs = inputs.to(self.device)                        
            outputs = self.model(inputs)            

            loss = 0
            
            if self.data == 'breastmnist':
                targets = one_hot(targets, num_classes=2)
            targets = torch.squeeze(targets, 1)

            data_hist = np.zeros(self.n_classes)
            for target in targets:
                ind = np.where(target==1)
                data_hist[ind] += 1
            data_hist /= self.train_loader.batch_size
            ce_weights = torch.Tensor(data_hist).to(self.device)
            criterion = nn.BCEWithLogitsLoss(weight=ce_weights)            
            targets = targets.float()
            targets = targets.to(self.device)

            if self.wceloss:
                wceloss = criterion(outputs, targets)
                loss = loss + wceloss
                # print(f'{loss.size()}, {wceloss.size()}')
                # print(f'{type(loss)}, {type(wceloss)}')
            if self.training_type=='dam':
                damLossVal = self.damLoss(outputs, targets)
                loss = loss + damLossVal
                # print(f'{loss.size()}, {damLossVal.size()}')
                # print(f'{type(loss[0])}, {type(damLossVal[0])}')
                # print(f'{loss}, {damLossVal}')
            if self.focalloss:
                focalloss = sigmoid_focal_loss(outputs, targets, self.alpha, self.gamma)
                loss = loss + focalloss

            if self.cdloss or self.scloss:
                if self.training_type == 'vgg19':
                    gradcam = LayerGradCam(self.model, layer=self.model.features[34])
                elif self.training_type == 'resnet101':
                    gradcam = LayerGradCam(self.model, layer=self.model.layer4[2].conv3)
                else:
                    gradcam = LayerGradCam(self.model, layer=self.model.features.denseblock4.denselayer16.conv2)
                attr_classes = [torch.Tensor(gradcam.attribute(inputs=inputs, target = [i] * inputs.shape[0])).to(self.device) for i in range(self.n_classes)]

                if self.cdloss:
                    cdcriterion = ClassDistinctivenessLoss(device=self.device)
                    cdloss_value = cdcriterion(attr_classes)
                    cdloss_value = self.cdloss_weight * cdloss_value
                    loss = loss + cdloss_value

                if self.scloss:
                    upsampled_attr_val = [LayerAttribution.interpolate(attr, inputs[0].shape[-2:]) for
                                        attr in attr_classes]
                    sccriterion = SpatialCoherenceConv(device=self.device, kernel_size=9)
                    scloss_value = sccriterion(upsampled_attr_val, device=self.device)
                    scloss_value = self.scloss_weight * scloss_value
                    loss = loss + scloss_value
                    
            predictions = self.sigmoid(outputs).to(self.device)
            y_pred = torch.cat((y_pred, predictions), 0)
            y_true = torch.cat((y_true, targets), 0)
            losses.append(loss.item())
            
            loss.backward()
            self.optimizer.step()

        self.train_losses.append(sum(losses) / len(losses))
        self.train_aucs.append(self.metrics(y_pred, y_true))

    def evaluate(self):
        #self.model.eval()

        losses = []
        y_true = torch.tensor([]).to(self.device)
        y_pred = torch.tensor([]).to(self.device)

        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)

                loss = 0 

                if self.data == 'breastmnist':
                    targets = one_hot(targets, num_classes=2)
                    
                targets = torch.squeeze(targets, 1)

                data_hist = np.zeros(self.n_classes)
                for target in targets:
                    ind = np.where(target==1)
                    data_hist[ind] += 1
                data_hist /= self.train_loader.batch_size
                ce_weights = torch.Tensor(data_hist).to(self.device)
                criterion = nn.BCEWithLogitsLoss(weight=ce_weights)
                # damLoss = AUCM_MultiLabel(num_classes=5)             
                targets = targets.float()
                targets = targets.to(self.device)

                if self.wceloss:
                    wceloss = criterion(outputs, targets)
                    loss = loss + wceloss
                if self.training_type=='dam':
                    damLossVal = self.damLoss(outputs, targets)
                    loss = loss + damLossVal

                if self.focalloss:
                    focalloss = sigmoid_focal_loss(outputs, targets, self.alpha, self.gamma)
                    loss = loss + focalloss

                if self.cdloss or self.scloss:
                    if self.training_type == 'vgg19':
                        gradcam = LayerGradCam(self.model, layer=self.model.features[34])
                    elif self.training_type == 'resnet101':
                        gradcam = LayerGradCam(self.model, layer=self.model.layer4[2].conv3)
                    else:
                        gradcam = LayerGradCam(self.model, layer=self.model.features.denseblock4.denselayer16.conv2)
                    attr_classes = [torch.Tensor(gradcam.attribute(inputs=inputs, target = [i] * inputs.shape[0])).to(self.device) for i in range(self.n_classes)]

                    if self.cdloss:
                        cdcriterion = ClassDistinctivenessLoss(device=self.device)
                        cdloss_value = cdcriterion(attr_classes)
                        cdloss_value = self.cdloss_weight * cdloss_value
                        loss = loss + cdloss_value

                    if self.scloss:
                        upsampled_attr_val = [LayerAttribution.interpolate(attr, inputs[0].shape[-2:]) for
                                            attr in attr_classes]
                        sccriterion = SpatialCoherenceConv(device=self.device, kernel_size=9)
                        scloss_value = sccriterion(upsampled_attr_val, device=self.device)
                        scloss_value = self.scloss_weight * scloss_value
                        loss = loss + scloss_value
                
                predictions = self.sigmoid(outputs).to(self.device)

                y_pred = torch.cat((y_pred, predictions), 0)
                y_true = torch.cat((y_true, targets), 0)
                losses.append(loss.item())

        self.val_losses.append(sum(losses) / len(losses))
        self.val_aucs.append(self.metrics(y_pred, y_true))
        ece_list = []
        for i in range(self.n_classes):
            ece_list.append(self.ece(y_pred[:,i], y_true[:,i]).cpu().detach().numpy())
        self.ece_vals.append(torch.from_numpy(np.array(ece_list)))

    def train(self, train_loader, val_loader, epochs):
        if self.training_type=='dam':            
            self.wceloss=False       
            self.damLoss = AUCM_MultiLabel(num_classes=5)            
            lr = 0.1 
            epoch_decay = 2e-3
            weight_decay = 1e-5
            margin = 1.0
            self.optimizer = PESG(self.model, 
                 loss_fn=self.damLoss,
                 lr=lr, 
                 margin=margin, 
                 epoch_decay=epoch_decay, 
                 weight_decay=weight_decay)        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        best_val_auc = 0

        for epoch in trange(self.epochs):
            if epoch > 0:                
                self.optimizer.update_regularizer(decay_factor = 10)
            self.train_one_epoch()
            self.evaluate()
            
            print("Epoch {0}: Training Loss = {1}, Validation Loss = {2}, Average Training AUC = {3}, Average Validation AUC = {4}, Average Calibration Error = {5}"
            .format(epoch+1, self.train_losses[-1], self.val_losses[-1], 
            sum(self.train_aucs[-1])/self.n_classes, sum(self.val_aucs[-1])/self.n_classes, sum(self.ece_vals[-1])/self.n_classes))

            losses = [self.train_losses[-1], self.val_losses[-1]]
            self.loss_df.loc['Epoch {}'.format(epoch+1)] = np.array(losses)

            train_auc = self.train_aucs[-1]
            train_auc = train_auc.tolist()
            train_auc.append(sum(train_auc)/len(train_auc))
            self.train_auc_df.loc['Epoch {}'.format(epoch+1)] = train_auc

            ece_val = self.ece_vals[-1]
            ece_val = ece_val.tolist()
            ece_val.append(sum(ece_val)/len(ece_val))
            self.ece_val_df.loc['Epoch {}'.format(epoch+1)] = ece_val

            val_auc = self.val_aucs[-1]
            val_auc = val_auc.tolist()
            val_auc.append(sum(val_auc)/len(val_auc))
            self.val_auc_df.loc['Epoch {}'.format(epoch+1)] = val_auc
            best_val_auc = max(best_val_auc, sum(val_auc)/len(val_auc))

            self.early_stopping(self.val_losses[-1], self.model, epoch)
            """
            if self.early_stopping.early_stop:
                print("Early stopping")
                break
            """

        time = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))
        time = str(time.day) + '_' + str(time.month) + '_' + str(time.year) + '_' + str(time.hour) + '_' + str(time.minute) + '_' + str(time.second)

        if self.grid_search==False:
            if self.training_type=='double-stage':
                self.loss_df.to_csv('results/{}_loss_double_stage_{}.csv'.format(self.data, time))
                self.train_auc_df.to_csv('results/{}_train_auc_double_stage_{}.csv'.format(self.data, time))
                self.val_auc_df.to_csv('results/{}_val_auc_double_stage_{}.csv'.format(self.data, time))
            elif self.training_type=='dam':
                self.loss_df.to_csv('results/{}_loss_dam_{}.csv'.format(self.data, time))
                self.train_auc_df.to_csv('results/{}_train_auc_dam_{}.csv'.format(self.data, time))
                self.val_auc_df.to_csv('results/{}_val_auc_dam_{}.csv'.format(self.data, time))                
            elif self.training_type=='vgg19':
                if not self.cdloss:
                    self.loss_df.to_csv('results/{}_loss_vgg19_{}.csv'.format(self.data, time))
                    self.train_auc_df.to_csv('results/{}_train_auc_vgg19_{}.csv'.format(self.data, time))
                    self.val_auc_df.to_csv('results/{}_val_auc_vgg19_{}.csv'.format(self.data, time))
                else:
                    self.loss_df.to_csv('results/{}_vgg19_loss_wce_cd_{}.csv'.format(self.data, time))
                    self.train_auc_df.to_csv('results/{}_vgg19_train_auc_wce_cd_{}.csv'.format(self.data, time))
                    self.val_auc_df.to_csv('results/{}_vgg19_val_auc_wce_cd_{}.csv'.format(self.data, time))                    
            elif self.training_type=='resnet101':
                if not self.cdloss:
                    self.loss_df.to_csv('results/{}_loss_resnet101_{}.csv'.format(self.data, time))
                    self.train_auc_df.to_csv('results/{}_train_auc_resnet101_{}.csv'.format(self.data, time))
                    self.val_auc_df.to_csv('results/{}_val_auc_resnet101_{}.csv'.format(self.data, time))
                else:
                    self.loss_df.to_csv('results/{}_resnet101_loss_wce_cd_{}.csv'.format(self.data, time))
                    self.train_auc_df.to_csv('results/{}_resnet101_train_auc_wce_cd_{}.csv'.format(self.data, time))
                    self.val_auc_df.to_csv('results/{}_resnet101_val_auc_wce_cd_{}.csv'.format(self.data, time))
            elif self.cdloss and self.scloss:
                self.loss_df.to_csv('results/{}_loss_wce_cd_sc_{}.csv'.format(self.data, time))
                self.train_auc_df.to_csv('results/{}_train_auc_wce_cd_sc_{}.csv'.format(self.data, time))
                self.val_auc_df.to_csv('results/{}_val_auc_wce_cd_sc_{}.csv'.format(self.data, time))
            elif self.cdloss:
                self.loss_df.to_csv('results/{}_loss_wce_cd_{}.csv'.format(self.data, time))
                self.train_auc_df.to_csv('results/{}_train_auc_wce_cd_{}.csv'.format(self.data, time))
                self.val_auc_df.to_csv('results/{}_val_auc_wce_cd_{}.csv'.format(self.data, time))
            elif self.scloss:
                self.loss_df.to_csv('results/{}_loss_wce_sc_{}.csv'.format(self.data, time))
                self.train_auc_df.to_csv('results/{}_train_auc_wce_sc_{}.csv'.format(self.data, time))
                self.val_auc_df.to_csv('results/{}_val_auc_wce_sc_{}.csv'.format(self.data, time))
            elif self.focalloss:
                self.loss_df.to_csv('results/{}_loss_focal_{}.csv'.format(self.data, time))
                self.train_auc_df.to_csv('results/{}_train_auc_focal_{}.csv'.format(self.data, time))
                self.val_auc_df.to_csv('results/{}_val_auc_focal_{}.csv'.format(self.data, time))
            else:
                self.loss_df.to_csv('results/{}_loss_wce_{}.csv'.format(self.data, time))
                self.train_auc_df.to_csv('results/{}_train_auc_wce_{}.csv'.format(self.data, time))
                self.val_auc_df.to_csv('results/{}_val_auc_wce_{}.csv'.format(self.data, time))
        else:
            self.loss_batch_df.to_csv('results/{}_{}_{}.csv'.format(self.data, self.cdloss_weight, time))
            self.train_auc_df.to_csv('results/{}_{}_train_auc_wce_{}.csv'.format(self.data, self.cdloss_weight, time))
            self.val_auc_df.to_csv('results/{}_{}_val_auc_wce_{}.csv'.format(self.data, self.cdloss_weight, time))
            with open("output.txt", "a") as f:
                print("Best val accuracy with cdloss weight = {} is {}".format(self.cdloss_weight, best_val_auc), file=f)
        self.ece_val_df.to_csv('results/{}_ece_val_{}_{}.csv'.format(self.data, self.lossfn, time))
        