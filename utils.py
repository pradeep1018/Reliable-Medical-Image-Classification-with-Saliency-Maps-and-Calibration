import torch
import numpy as np

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, args):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = args.patience
        self.verbose = args.verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.wceloss = args.wceloss
        self.focalloss = args.focalloss
        self.cdloss = args.cdloss
        self.scloss = args.scloss
        self.training_type = args.training_type
        self.data = args.data
        self.best_epoch = 0

    def __call__(self, val_loss, model, epoch):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self.best_epoch = epoch

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if self.training_type=='double-stage':        
            torch.save(model.state_dict(), 'model-weights/{}_double_stage.pt'.format(self.data))
        elif self.training_type=='dam':
            torch.save(model.state_dict(), 'model-weights/{}_dam.pt'.format(self.data))
        elif self.training_type=='vgg19':
            if self.cdloss:
                torch.save(model.state_dict(), 'model-weights/{}_vgg19_cd.pt'.format(self.data))
            else:
                torch.save(model.state_dict(), 'model-weights/{}_vgg19.pt'.format(self.data))
        elif self.training_type=='resnet101':            
            if self.cdloss:
                torch.save(model.state_dict(), 'model-weights/{}_resnet101_cd.pt'.format(self.data))
            else:
                torch.save(model.state_dict(), 'model-weights/{}_resnet101.pt'.format(self.data))            
        elif self.cdloss and self.scloss:
            torch.save(model.state_dict(), 'model-weights/{}_wce_cd_sc.pt'.format(self.data))
        elif self.cdloss:
            torch.save(model.state_dict(), 'model-weights/{}_wce_cd.pt'.format(self.data))
        elif self.scloss:
            torch.save(model.state_dict(), 'model-weights/{}_wce_sc.pt'.format(self.data))
        elif self.focalloss:
            torch.save(model.state_dict(), 'model-weights/{}_focal.pt'.format(self.data))
        else:
            torch.save(model.state_dict(), 'model-weights/{}_wce.pt'.format(self.data))
        self.val_loss_min = val_loss

def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False