import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch import optim
from torch.nn.functional import one_hot
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from captum.attr import LayerGradCam, LayerAttribution, LRP
from tqdm import tqdm, trange
from torchvision.models import densenet121
import torchvision.models as models
import datetime
import pytz
from libauc.losses import AUCM_MultiLabel
from libauc.optimizers import PESG
from libauc.models import densenet121 as DenseNet121
from torchmetrics.classification import BinaryCalibrationError
from sklearn.calibration import calibration_curve
import calibration as cal

import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import trange
import argparse

from dataset import CheXpertData, get_data_medmnist
from train import Trainer
from utils import str_to_bool
import numpy as np

from losses import ClassDistinctivenessLoss, SpatialCoherenceConv, sigmoid_focal_loss
from metrics import AUC
from utils import EarlyStopping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_classes = 5
damlossfn = AUCM_MultiLabel(num_classes=5)
sigmoid = nn.Sigmoid()
metrics = AUC()
ece = BinaryCalibrationError()

valid_csv = '../BTP/CheXpert-v1.0-small/valid.csv'
val_data = CheXpertData(valid_csv, mode='val')
val_loader = DataLoader(val_data,
                            drop_last=True,shuffle=False,
                            batch_size=1, num_workers=32, pin_memory=True)

model = densenet121(weights='DEFAULT')
model.classifier = nn.Linear(1024, n_classes)
model.load_state_dict(torch.load('model-weights/chexpert_dam.pt'))
model = model.to(device)

class BaseCalibrator:
    """ Abstract calibrator class
    """
    def __init__(self):
        self.n_classes = None

    def fit(self, logits, y):
        raise NotImplementedError

    def calibrate(self, probs):
        raise NotImplementedError


class TSCalibrator(BaseCalibrator):
    """ Maximum likelihood temperature scaling (Guo et al., 2017)
    """

    def __init__(self, temperature=1.):
        super().__init__()
        self.temperature = temperature

        self.loss_trace = None

    def fit(self, logits, y):
        """ Fits temperature scaling using hard labels.
        """
        # Pre-processing
        self.n_classes = logits.shape[1]
        _model_logits = logits
        _y = y
        _temperature = torch.tensor(self.temperature, requires_grad=True)

        # Optimization parameters
        nll = nn.CrossEntropyLoss()  # Supervised hard-label loss
        num_steps = 7500
        learning_rate = 0.05
        grad_tol = 1e-3  # Gradient tolerance for early stopping
        min_temp, max_temp = 1e-2, 1e4  # Upper / lower bounds on temperature

        optimizer = optim.Adam([_temperature], lr=learning_rate)

        loss_trace = []  # Track loss over iterations
        step = 0
        converged = False
        while not converged:

            optimizer.zero_grad()
            loss = nll(_model_logits / _temperature, _y)
            loss.backward()
            optimizer.step()
            loss_trace.append(loss.item())

            with torch.no_grad():
                _temperature.clamp_(min=min_temp, max=max_temp)

            step += 1
            if step > num_steps:
                warnings.warn('Maximum number of steps reached -- may not have converged (TS)')
            converged = (step > num_steps) or (np.abs(_temperature.grad) < grad_tol)

        self.loss_trace = loss_trace
        self.temperature = _temperature.item()

    def calibrate(self, probs):
        calibrated_probs = probs ** (1. / self.temperature)  
        calibrated_probs /= torch.sum(calibrated_probs, axis=1, keepdims=True)  # Normalize
        return calibrated_probs

class IMaxCalibrator(BaseCalibrator):
    """ I-Max Binning calibration (Patel et al., 2021)
    https://arxiv.org/pdf/2006.13092.pdf
    """

    def __init__(self, mode='CW', num_bins=15):
        super().__init__()
        # mode in ['cw', 'sCW', 'top1']
        self.cfg = io.AttrDict(dict(
            # All
            cal_setting=mode,  # CW, sCW or top1  # CW seems to be much better than sCW
            num_bins=num_bins,
            # Binning
            Q_method="imax",
            Q_binning_stage="raw",  # bin the raw logodds or the 'scaled' logodds
            Q_binning_repr_scheme="sample_based",
            Q_bin_repr_during_optim="pred_prob_based",
            Q_rnd_seed=928163,
            Q_init_mode="kmeans"
        ))
        self.calibrator = None

    def calibrate(self, probs):
        logits = np.log(np.clip(probs, 1e-50, 1))
        logodds = imax_utils.quick_logits_to_logodds(logits, probs=probs)
        cal_logits, cal_logodds, cal_probs, assigned = self.calibrator(logits, logodds)
        return cal_probs

    def fit(self, logits, y):
        n_samples, n_classes = logits.shape
        self.n_classes = n_classes
        self.cfg['n_classes'] = n_classes
        # y must be one-hot
        if y.ndim == 1:
            y_onehot = np.eye(self.n_classes)[y]
        else:
            y_onehot = y

        logodds = imax_utils.quick_logits_to_logodds(logits)
        self.calibrator = imax_calibration.learn_calibrator(self.cfg,
                                                            logits=logits,
                                                            logodds=logodds,
                                                            y=y_onehot)

class EnsembleTSCalibrator(BaseCalibrator):
    """ Ensemble Temperature Scaling (Zhang et al., 2020)
    This is just a thin wrapper around ensemble_ts.py for convenience.
    """

    def __init__(self, temperature=1.):
        super().__init__()
        self.temperature = temperature
        self.weights = None

    def calibrate(self, probs):
        p1 = probs.cpu().detach().numpy()
        tempered_probs = probs.cpu().detach().numpy() ** (1. / self.temperature)  # Temper
        tempered_probs /= np.sum(tempered_probs, axis=1, keepdims=True)  # Normalize
        p0 = tempered_probs
        p2 = np.ones_like(p0) / self.n_classes

        calibrated_probs = self.weights[0] * p0 + self.weights[1] * p1 + self.weights[2] * p2

        return calibrated_probs

    def fit(self, logits, y):
        from ensemble_ts import ets_calibrate
        self.n_classes = logits.shape[1]

        #y = y.cpu().detach().numpy()
        # labels need to be one-hot for ETS
        #_y = np.eye(self.n_classes)[y]

        t, w = ets_calibrate(logits.cpu().detach().numpy(), y.cpu().detach().numpy(), self.n_classes, loss='mse')  # loss = 'ce'
        self.temperature = t
        self.weights = w

def evaluate():
    model.eval()
    losses = []
    y_true = torch.tensor([]).to(device)
    y_pred = torch.tensor([]).to(device)
    with torch.no_grad():
        for itr, (inputs, targets) in enumerate(tqdm(val_loader)):
            inputs = inputs.to(device)
            outputs = model(inputs)

            loss = 0
            targets = torch.squeeze(targets, 1)
            data_hist = np.zeros(n_classes)
            for target in targets:
                ind = np.where(target==1)
                data_hist[ind] += 1
            data_hist /= val_loader.batch_size
            ce_weights = torch.Tensor(data_hist).to(device)        
            targets = targets.float()
            targets = targets.to(device)
            
            predictions = sigmoid(outputs).to(device)
            y_pred = torch.cat((y_pred, predictions), 0)
            y_true = torch.cat((y_true, targets), 0)

    val_auc = metrics(y_pred, y_true)
    ece_list = []
    for i in range(n_classes):
        ece_list.append(ece(y_pred[:,i], y_true[:,i]).cpu().detach().numpy())
    ece_val = torch.from_numpy(np.array(ece_list))
    print(val_auc)
    print(ece_val)
    print(sum(val_auc)/len(val_auc))
    print(sum(ece_val)/len(ece_val))

    calibrator = TSCalibrator()
    model_probs = torch.clip(y_pred, 1e-50, 1)
    model_logits = torch.log(y_pred)
    calibrator.fit(model_logits, y_true)
    #calibrated_probs = torch.from_numpy(calibrator.calibrate(y_pred)).to(device)
    calibrated_probs = calibrator.calibrate((y_pred))
    ece_list = []
    for i in range(n_classes):
        ece_list.append(ece(calibrated_probs[:,i], y_true[:,i]).cpu().detach().numpy())
    ece_val = torch.from_numpy(np.array(ece_list))
    print(ece_val)
    print(sum(ece_val)/len(ece_val))
    val_auc = metrics(calibrated_probs, y_true)
    print(val_auc)
    print(sum(val_auc)/len(val_auc))
    x1, y1 = calibration_curve(y_true[:,4].cpu().detach().numpy(), y_pred[:,4].cpu().detach().numpy(), n_bins = 15, normalize=True)
    x2, y2 = calibration_curve(y_true[:,4].cpu().detach().numpy(), calibrated_probs[:,4].cpu().detach().numpy(), n_bins = 15, normalize=True)

    plt.plot([0, 1], [0, 1], linestyle = '--', label = 'Ideally Calibrated')
 
    # Plot model's calibration curve
    plt.plot(y1, x1, marker = '.', label = 'Uncalibrated model')
    plt.plot(y2, x2, marker = '.', label = 'Calibrated model')
    
    leg = plt.legend(loc = 'upper left')
    plt.xlabel('Average Predicted Probability in each bin')
    plt.ylabel('Ratio of positives')
    plt.savefig('img.png')

evaluate()