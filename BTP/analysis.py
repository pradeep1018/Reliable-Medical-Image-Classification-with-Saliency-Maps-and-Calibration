import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.models import densenet121
from captum.attr import LayerGradCam, LayerAttribution
import pandas as pd

from losses import ClassDistinctivenessLoss, SpatialCoherenceConv
from metrics import AUC
from dataset import CheXpertData

"""
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_classes = 5
valid_csv = 'CheXpert-v1.0-small/valid.csv'
val_data = CheXpertData(valid_csv, mode='val')

val_loader = DataLoader(val_data,
                            drop_last=True,shuffle=False,
                            batch_size=8, num_workers=32, pin_memory=True)

model_wce = densenet121()
model_wce.classifier = nn.Linear(1024, n_classes)
model_wce.load_state_dict(torch.load('model-weights/chexpert_wce.pt'))
model_wce = model_wce.to(device)

model_wce_cd_sc = densenet121()
model_wce_cd_sc.classifier = nn.Linear(1024, n_classes)
model_wce_cd_sc.load_state_dict(torch.load('model-weights/chexpert_wce_cd_sc.pt'))
model_wce_cd_sc = model_wce_cd_sc.to(device)

model_double_stage = densenet121()
model_double_stage.classifier = nn.Linear(1024, n_classes)
model_double_stage.load_state_dict(torch.load('model-weights/chexpert_double_stage.pt'))
model_double_stage = model_double_stage.to(device)

sigmoid = nn.Sigmoid()
metrics = AUC()

def test(model, cdloss=False, scloss=False, cdloss_weight=1.2, scloss_weight=0.9):
    gradcam = LayerGradCam(model, layer=model.features[-1])

    model.eval()

    y_true = torch.tensor([]).to(device)
    y_pred = torch.tensor([]).to(device)

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            targets = targets.to(device)
            
            predictions = sigmoid(outputs).to(device)

            y_pred = torch.cat((y_pred, predictions), 0)
            y_true = torch.cat((y_true, targets), 0)

    return metrics(y_pred, y_true)

auc_wce = test(model_wce)  
auc_wce = auc_wce.tolist()
auc_wce.append(sum(auc_wce)/len(auc_wce))

auc_wce_cd_sc = test(model_wce_cd_sc)
auc_wce_cd_sc = auc_wce_cd_sc.tolist()
auc_wce_cd_sc.append(sum(auc_wce_cd_sc)/len(auc_wce_cd_sc))

auc_double_stage = test(model_double_stage)
auc_double_stage = auc_double_stage.tolist()
auc_double_stage.append(sum(auc_double_stage)/len(auc_double_stage))

print('AUC of wce model is',auc_wce[:-1])
print('AUC of custom loss model is',auc_wce_cd_sc[:-1])
print('AUC of double stage model is',auc_double_stage[:-1])

print('Mean AUC of wce model is',auc_wce[-1])
print('Mean AUC of custom loss model is',auc_wce_cd_sc[-1])
print('Mean AUC of double stage model is',auc_double_stage[-1])
"""
"""
df1 = pd.read_csv('results/chexpert_loss_wce_11_3_2023_16_7_27.csv')
df2 = pd.read_csv('results/chexpert_loss_wce_cd_sc_11_3_2023_19_13_16.csv')
df3 = pd.read_csv('results/chexpert_loss_double_stage_12_3_2023_3_7_27.csv')

plt.plot(df1['Training Loss'], label='cross-entropy loss')
plt.plot(df2['Training Loss'], label='custom loss')
plt.plot(df3['Training Loss'], label='modified pipeline')
plt.title('Training loss')
plt.legend()
plt.savefig('results/train_loss.png')
plt.close()

plt.plot(df1['Validation Loss'], label='cross-entropy loss')
plt.plot(df2['Validation Loss'], label='custom loss')
plt.plot(df3['Validation Loss'], label='modified pipeline')
plt.title('Validation loss')
plt.legend()
plt.savefig('results/val_loss.png')
plt.close()

df4 = pd.read_csv('results/chexpert_train_auc_wce_11_3_2023_16_7_27.csv')
df5 = pd.read_csv('results/chexpert_train_auc_wce_cd_sc_11_3_2023_19_13_16.csv')
df6 = pd.read_csv('results/chexpert_train_auc_double_stage_12_3_2023_3_7_27.csv')
plt.plot(df4['Mean'], label='cross-entropy loss')
plt.plot(df5['Mean'], label='custom loss')
plt.plot(df6['Mean'], label='modified pipeline')
plt.title('Mean Training AUC')
plt.legend()
plt.savefig('results/mean_train_auc.png')
plt.close()

df7 = pd.read_csv('results/chexpert_val_auc_wce_11_3_2023_16_7_27.csv')
df8 = pd.read_csv('results/chexpert_val_auc_wce_cd_sc_11_3_2023_19_13_16.csv')
df9 = pd.read_csv('results/chexpert_val_auc_double_stage_12_3_2023_3_7_27.csv')
plt.plot(df7['Mean'], label='cross-entropy loss')
plt.plot(df8['Mean'], label='custom loss')
plt.plot(df9['Mean'], label='modified pipeline')
plt.title('Mean Validation AUC')
plt.legend()
plt.savefig('results/mean_val_auc.png')
plt.close()
"""

df1 = pd.read_csv('results/chexpert_loss_diff_16_3_2023_16_28_53.csv')
df2 = pd.read_csv('results/chexpert_loss_diff_16_3_2023_16_32_24.csv')
plt.plot(df1['custom loss'],label='custom loss 1')
plt.plot(df2['custom loss'],label='custom loss 2')
plt.plot(df1['cross entropy'],label='custom loss 1 ce')
plt.plot(df2['cross entropy'],label='custom loss 2 ce')
plt.plot(df1['class distinctiveness'],label='custom loss 1 cd')
plt.plot(df2['class distinctiveness'],label='custom loss 2 cd')
plt.plot(df1['spatial coherence'],label='custom loss 1 sc')
plt.plot(df2['spatial coherence'],label='custom loss 2 sc')
plt.legend()
plt.savefig('hi.png')