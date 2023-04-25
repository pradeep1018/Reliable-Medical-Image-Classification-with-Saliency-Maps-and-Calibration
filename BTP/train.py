import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from captum.attr import LayerGradCam, LayerAttribution
from tqdm import tqdm, trange
from torchvision.models import densenet121
import datetime
import pytz

from losses import ClassDistinctivenessLoss, SpatialCoherenceConv
from metrics import AUC
from utils import EarlyStopping

class CheXpert:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_classes = args.n_classes
        self.batch_size = args.batch_size
        self.data = args.data
        self.training_type = args.training_type
        self.cdloss = args.cdloss
        self.cdloss_weight = args.cdloss_weight
        self.scloss = args.scloss
        self.scloss_weight = args.scloss_weight

        if self.data=='chexpert':
            if self.training_type == 'double-stage':
                self.model = densenet121()
                self.model.classifier = nn.Linear(1024, self.n_classes)
                self.model.load_state_dict(torch.load('model-weights/chexpert_wce.pt'))
            else:
                self.model = densenet121(weights='DEFAULT')
                self.model.classifier = nn.Linear(1024, self.n_classes)
        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        self.sigmoid = nn.Sigmoid()
        #self.gradcam = LayerGradCam(self.model, layer=self.model.features[-1])
        #self.lrp = LRP(self.model)
        #self.integrated_gradients = IntegratedGradients(self.model)
        #self.saliency = Saliency(self.model)

        self.metrics = AUC()
        self.early_stopping = EarlyStopping(args)

        self.iteration = 0
        self.wceloss_scale = 1
        self.scloss_scale = 1
        self.cdloss_scale = 1

        self.train_losses = []
        self.val_losses = []
        self.train_aucs = []
        self.val_aucs = []

        self.loss_df = pd.DataFrame(columns=['Training Loss','Validation Loss'])
        if args.data=='chexpert':
            self.train_auc_df = pd.DataFrame(columns=['Atelectasis','Cardiomegaly','Consolidation','Edema','Plural Effusion','Mean'])
            self.val_auc_df = pd.DataFrame(columns=['Atelectasis','Cardiomegaly','Consolidation','Edema','Plural Effusion','Mean'])
        if self.cdloss or self.scloss:
            self.loss_diff_df = pd.DataFrame(columns=['custom loss','cross entropy','class distinctiveness','spatial coherence'])

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

            data_hist = np.zeros(self.n_classes)
            for target in targets:
                ind = np.where(target==1)
                data_hist[ind] += 1
            data_hist /= self.train_loader.batch_size

            targets = targets.to(self.device)
            ce_weights = torch.Tensor(data_hist).to(self.device)
            criterion = nn.BCEWithLogitsLoss(weight=ce_weights)

            wceloss = criterion(outputs, targets)
            if self.iteration == 0:
                self.wceloss_scale = 1 / (wceloss.item())
            wceloss *= self.wceloss_scale
            loss = loss + wceloss

            if self.cdloss or self.scloss:
                gradcam = LayerGradCam(self.model, layer=self.model.features[-1])
                attr_classes = [torch.Tensor(gradcam.attribute(inputs=inputs, target = [i] * inputs.shape[0])).to(self.device) for i in range(self.n_classes)]
                
                loss_val = [0, 0, 0, 0]                
                loss_val[1] = wceloss.item()

                if self.cdloss:
                    cdcriterion = ClassDistinctivenessLoss(device=self.device)
                    cdloss_value = cdcriterion(attr_classes)
                    if self.iteration == 0:
                        self.cdloss_scale = 1 / (cdloss_value.item())
                    cdloss_value *= self.cdloss_scale
                    loss = loss + self.cdloss_weight * cdloss_value
                    loss_val[2] = cdloss_value.item()

                if self.scloss:
                    upsampled_attr_val = [LayerAttribution.interpolate(attr, inputs[0].shape[-2:]) for
                                        attr in attr_classes]
                    sccriterion = SpatialCoherenceConv(device=self.device, kernel_size=9)
                    scloss_value = sccriterion(upsampled_attr_val, device=self.device)
                    if self.iteration == 0:
                        self.scloss_scale = 1 / (scloss_value.item())
                    scloss_value *= self.scloss_scale
                    loss = loss + self.scloss_weight * scloss_value
                    loss_val[3] = scloss_value.item()

                # loss_val = [loss.item(), wceloss.item(), cdloss_value.item(), scloss_value.item()]
                loss_val[0] = loss.item()
                self.loss_diff_df.loc[len(self.loss_diff_df.index)] = loss_val
                    
            predictions = self.sigmoid(outputs).to(self.device)
            y_pred = torch.cat((y_pred, predictions), 0)
            y_true = torch.cat((y_true, targets), 0)
            losses.append(loss.item())
            
            loss.backward()
            self.optimizer.step()

            self.iteration += 1

        self.train_losses.append(sum(losses) / len(losses))
        self.train_aucs.append(self.metrics(y_pred, y_true))

    def evaluate(self):
        self.model.eval()

        losses = []
        y_true = torch.tensor([]).to(self.device)
        y_pred = torch.tensor([]).to(self.device)

        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)

                loss = 0 

                data_hist = np.zeros(self.n_classes)
                for target in targets:
                    ind = np.where(target==1)
                    data_hist[ind] += 1
                data_hist /= self.val_loader.batch_size

                targets = targets.to(self.device)
                ce_weights = torch.Tensor(data_hist).to(self.device)
                criterion = nn.BCEWithLogitsLoss(weight=ce_weights)

                wceloss = criterion(outputs, targets)
                wceloss *= self.wceloss_scale
                loss = loss + wceloss

                if self.cdloss or self.scloss:
                    gradcam = LayerGradCam(self.model, layer=self.model.features[-1])
                    attr_classes = [torch.Tensor(gradcam.attribute(inputs=inputs, target = [i] * inputs.shape[0])).to(self.device) for i in range(self.n_classes)]

                    if self.cdloss:
                        cdcriterion = ClassDistinctivenessLoss(device=self.device)
                        cdloss_value = cdcriterion(attr_classes)
                        cdloss_value *= self.cdloss_scale
                        loss = loss + self.cdloss_weight * cdloss_value

                    if self.scloss:
                        upsampled_attr_val = [LayerAttribution.interpolate(attr, inputs[0].shape[-2:]) for
                                            attr in attr_classes]
                        sccriterion = SpatialCoherenceConv(device=self.device, kernel_size=9)
                        scloss_value = sccriterion(upsampled_attr_val, device=self.device)
                        scloss_value *= self.scloss_scale
                        loss = loss + self.scloss_weight * scloss_value
                
                predictions = self.sigmoid(outputs).to(self.device)

                y_pred = torch.cat((y_pred, predictions), 0)
                y_true = torch.cat((y_true, targets), 0)
                losses.append(loss.item())

        self.val_losses.append(sum(losses) / len(losses))
        self.val_aucs.append(self.metrics(y_pred, y_true))

    def train(self, train_loader, val_loader, epochs):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs

        for epoch in trange(self.epochs):
            self.train_one_epoch()
            self.evaluate()
            
            print("Epoch {0}: Training Loss = {1}, Validation Loss = {2}, Average Training AUC = {3}, Average Validation AUC = {4}".
            format(epoch+1, self.train_losses[-1], self.val_losses[-1], 
            sum(self.train_aucs[-1])/self.n_classes, sum(self.val_aucs[-1])/self.n_classes))

            losses = [self.train_losses[-1], self.val_losses[-1]]
            self.loss_df.loc['Epoch {}'.format(epoch+1)] = losses

            train_auc = self.train_aucs[-1]
            train_auc = train_auc.tolist()
            train_auc.append(sum(train_auc)/len(train_auc))
            self.train_auc_df.loc['Epoch {}'.format(epoch+1)] = train_auc

            val_auc = self.val_aucs[-1]
            val_auc = val_auc.tolist()
            val_auc.append(sum(val_auc)/len(val_auc))
            self.val_auc_df.loc['Epoch {}'.format(epoch+1)] = val_auc

            self.early_stopping(self.val_losses[-1], self.model, epoch)
            """
            if self.early_stopping.early_stop:
                print("Early stopping")
                break
            """

        time = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))
        time = str(time.day) + '_' + str(time.month) + '_' + str(time.year) + '_' + str(time.hour) + '_' + str(time.minute) + '_' + str(time.second)

        if self.training_type=='double-stage':
            self.loss_df.to_csv('results/{}_loss_double_stage_{}.csv'.format(self.data, time))
            self.train_auc_df.to_csv('results/{}_train_auc_double_stage_{}.csv'.format(self.data, time))
            self.val_auc_df.to_csv('results/{}_val_auc_double_stage_{}.csv'.format(self.data, time))
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
        else:
            self.loss_df.to_csv('results/{}_loss_wce_{}.csv'.format(self.data, time))
            self.train_auc_df.to_csv('results/{}_train_auc_wce_{}.csv'.format(self.data, time))
            self.val_auc_df.to_csv('results/{}_val_auc_wce_{}.csv'.format(self.data, time))

        if self.cdloss or self.scloss:
            self.loss_diff_df.to_csv('results/{}_loss_diff_{}.csv'.format(self.data, time))
        