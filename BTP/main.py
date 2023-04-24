import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import trange
import argparse


from dataset import CheXpertData
from train import CheXpert
from utils import str_to_bool

def main(args):
    torch.manual_seed(0)

    if args.data=='chexpert':
        train_csv = 'CheXpert-v1.0-small/train.csv'
        valid_csv = 'CheXpert-v1.0-small/valid.csv'
        train_data = CheXpertData(train_csv, mode='train')
        val_data = CheXpertData(valid_csv, mode='val')

    train_data_subset = Subset(train_data, range(0, 10000))
    train_loader = DataLoader(train_data_subset,
                            drop_last=True,shuffle=True,
                            batch_size=args.batch_size, num_workers=32, pin_memory=True)
    val_loader = DataLoader(val_data,
                            drop_last=True,shuffle=False,
                            batch_size=args.batch_size, num_workers=32, pin_memory=True)

    if args.data=='chexpert':
        chexpert_model = CheXpert(args)
        chexpert_model.train(train_loader, val_loader, epochs=args.epochs)

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--data', default='chexpert', type=str)
    parser.add_argument('--training_type', default='single-stage', type=str)
    parser.add_argument('--cdloss', default='False', type=str)
    parser.add_argument('--cdloss_weight', default=1.2, type=int)
    parser.add_argument('--scloss', default='False', type=str)
    parser.add_argument('--scloss_weight', default=0.9, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--beta1', default=0.93, type=float)
    parser.add_argument('--beta2', default=0.999, type=float)
    parser.add_argument('--n_classes', default=5, type=int)
    parser.add_argument('--patience', default=7, type=int)
    parser.add_argument('--verbose', default='False', type=str)
    #grid search argument added
    parser.add_argument('--grid_search', default='False', type = str) 
    parser.add_argument('--cdloss_wtlist', default = '1,2,4,5,6,8', help = 'comma separated floats', type=str)
    args = parser.parse_args()
    args.grid_search = str_to_bool(args.grid_search)
    if args.grid_search == True:
        args.cdloss = True
        args.scloss = False        
        args.verbose = str_to_bool(args.verbose)
        wtlist = [int(wt) for wt in args.cdloss_wtlist.split(',')]  
        for wt in wtlist:
            args.cdloss_weight = wt
            # print(args)            
            main(args)
    else:        
        args.cdloss = str_to_bool(args.cdloss)
        args.scloss = str_to_bool(args.scloss)
        args.verbose = str_to_bool(args.verbose)
        # print(args)
        main(args)
    