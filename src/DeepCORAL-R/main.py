from __future__ import division
import argparse
import torch
from torch.utils import model_zoo
from torch.autograd import Variable
import os
import models
import utils
from data_loader import SugarDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import mean_squared_error
import numpy as np


parser = argparse.ArgumentParser(description='PyTorch DeepCORAL- ResNet101 experiment')
parser.add_argument('--backbone', type=str, default='resnet101', help="model backbone")
parser.add_argument('--substance', type=str, default='lacticacid', choices=['glucose', 'lacticacid'])

parser.add_argument('--dataset_dir', type=str, default=f'/mnt/beegfs/home/liu15/la-ood-tl-main/data/', help="source dataset file")
parser.add_argument('--save_dir', type=str, default='/mnt/beegfs/home/liu15/la-ood-tl-main/DeepCORAL/', help="save model firectory")

parser.add_argument('--gpu_id', type=str, nargs='?', default='3', help="device id to run")
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--epochs', type=int, default=300, help='Epochs')
parser.add_argument('--num_workers', type=int, default=8, help='num of workers')
parser.add_argument('--num_runs', type=int, default=3, help='num of experiment runs')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

args.source_path = args.dataset_dir + f'ss_{args.substance}.xlsx'
args.target_path = args.dataset_dir + f'cs_{args.substance}.xlsx'
args.save_dir = args.save_dir + f'runs_{args.substance}/'

print(f"Source Dataset: {args.source_path}")
print(f"Target Dataset: {args.target_path}")
print()


LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9


# source_loader = get_office31_dataloader(case='amazon', batch_size=BATCH_SIZE[0])
# target_loader = get_office31_dataloader(case='webcam', batch_size=BATCH_SIZE[1])


source_glucose_dataset = SugarDataset(data_path=args.source_path, source_domain=True, test_ratio=0.2)
target_glucose_dataset = SugarDataset(data_path=args.target_path, source_domain=False, test_ratio=0.2)

# Loading Train loader (source and target)
source_train_loader = DataLoader(source_glucose_dataset.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
target_train_loader = DataLoader(target_glucose_dataset.train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

# Loading Val loader (source and target)
source_val_loader = DataLoader(source_glucose_dataset.test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
target_val_loader = DataLoader(target_glucose_dataset.test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

# Loading whole dataset (target)
target_whole_dataset = SugarDataset(data_path=args.target_path, source_domain=False, test_ratio=0)
target_whole_loader = DataLoader(target_whole_dataset.train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


def train(model, optimizer, epoch, _lambda):

    total_loss_epoch = 0
    regression_loss_total = 0
    coral_loss_total = 0
    num_batches = 0

    model.train()
    # Expected size : xs -> (batch_size, 3, 300, 300), ys -> (batch_size)
    # source, target = list(enumerate(source_loader)), list(enumerate(target_loader))
    # train_steps = min(len(source), len(target))
    criterion_reg = nn.SmoothL1Loss().cuda()
    
    for batch_idx, (source_data, target_data) in enumerate(zip(source_train_loader, target_train_loader)):

        source_x, source_y_reg, source_domain = source_data
        target_x, target_y_reg, target_domain = target_data


        source_x, source_y_reg, source_domain = source_x.cuda(), source_y_reg.cuda(), source_domain.cuda() 
        target_x, target_y_reg, target_domain = target_x.cuda(), target_y_reg.cuda(), target_domain.cuda() 
            

        optimizer.zero_grad()
        
        source_feature, target_feature, source_out, target_out = model(source_x, target_x)

        regression_loss = criterion_reg(source_out.view(-1), source_y_reg)
        coral_loss = models.CORAL(source_feature, target_feature)

        sum_loss = _lambda*coral_loss + regression_loss
        sum_loss.backward()

        optimizer.step()
        
        regression_loss_total += regression_loss.item()
        coral_loss_total += coral_loss.item()
        total_loss_epoch += sum_loss.item()
        num_batches += 1

    print(f'[Epoch {epoch}]\tTotal Loss: {total_loss_epoch/num_batches:.4f}\tRegression Loss: {regression_loss_total/num_batches:.4f}\tCORAL Loss: {coral_loss_total/num_batches:.4f}\tLambda:{_lambda:.2f}')


def inference(model, target_test_loader):
    with torch.no_grad():
        model.eval()

        target_mse = 0
        num_batches = 0

        target_pred = []
        target_gt = []

        regression_criterion = nn.SmoothL1Loss().cuda()
        target_regression_loss = 0

        for target_data in target_test_loader:

            # Process source and target data
            target_x, target_y_reg, target_domain = target_data
            target_x, target_y_reg, target_domain = target_x.cuda(), target_y_reg.cuda(), target_domain.cuda()

            # Feature Extraction
            target_feature = model.sharedNet(target_x)

            # Regression Predictions
            target_pred_batch = model.regression_predictor(target_feature)
            target_pred += torch.squeeze(target_pred_batch)
            target_gt += target_y_reg
            
            # Compute loss for debuging
            target_regression_loss += regression_criterion(target_pred_batch.view(-1), target_y_reg).item()
            num_batches += 1

    target_pred = [tensor.to('cpu').item() for tensor in target_pred]
    target_gt = [tensor.to('cpu').item() for tensor in target_gt]
    target_mse = mean_squared_error(target_pred, target_gt)
    target_rmse = np.sqrt(target_mse)

    print(f'[The target set - {len(target_gt)} samples]\tTarget Regression Loss: {target_regression_loss/num_batches:.4f}\tMSE: {target_mse}\tRMSE: {target_rmse}\n')

    return target_mse, target_rmse


def validation(model, source_test_loader, target_test_loader):
    model.eval()

    source_pred = []
    target_pred = []

    source_gt = []
    target_gt = []

    num_batches = 0

    source_regression_loss = 0
    target_regression_loss = 0

    criterion_reg = nn.SmoothL1Loss().cuda()

    for batch_idx, (source_data, target_data) in enumerate(zip(source_test_loader, target_test_loader)):

        # 1. Process source and target data with same batch size
        real_batch_size = min(source_data[0].size(0), target_data[0].size(0))
        source_data = [data[:real_batch_size] for data in source_data]
        target_data = [data[:real_batch_size] for data in target_data]

        source_x, source_y_reg, source_domain = source_data
        target_x, target_y_reg, target_domain = target_data 

        source_x, source_y_reg, source_domain = source_x.cuda(), source_y_reg.cuda(), source_domain.cuda() 
        target_x, target_y_reg, target_domain = target_x.cuda(), target_y_reg.cuda(), target_domain.cuda() 


        source_feature, target_feature, source_pred_batch, target_pred_batch = model(source_x, target_x)

        source_pred += torch.squeeze(source_pred_batch)
        target_pred += torch.squeeze(target_pred_batch)
        source_gt += source_y_reg
        target_gt += target_y_reg

        source_regression_loss += criterion_reg(source_pred_batch.view(-1), source_y_reg).item()
        target_regression_loss += criterion_reg(target_pred_batch.view(-1), target_y_reg).item()
        num_batches += 1


    source_pred = [tensor.to('cpu').item() for tensor in source_pred]
    target_pred = [tensor.to('cpu').item() for tensor in target_pred]

    source_gt = [tensor.to('cpu').item() for tensor in source_gt]
    target_gt = [tensor.to('cpu').item() for tensor in target_gt]

    source_mse = mean_squared_error(source_pred, source_gt)
    target_mse = mean_squared_error(target_pred, target_gt)

    source_rmse = np.sqrt(source_mse)
    target_rmse = np.sqrt(target_mse)

    print(f"[Source Regression] - Regression Loss: {source_regression_loss/num_batches:.4f}\tMSE: {source_mse:.3f}\tRMSE: {source_rmse:.3f}\n"
      f"[Target Regression] - Regression Loss: {target_regression_loss/num_batches:.4f}\tMSE: {target_mse:.3f}\tRMSE: {target_rmse:.3f}")
    
    return target_rmse




# load AlexNet pre-trained model
def load_pretrained(model):
    url = 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
    pretrained_dict = model_zoo.load_url(url)
    model_dict = model.state_dict()

    # filter out unmatch dict and delete last fc bias, weight
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # del pretrained_dict['classifier.6.bias']
    # del pretrained_dict['classifier.6.weight']

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


if __name__ == '__main__':

    model = models.DeepCORAL_1D(backbone=args.backbone, substance=args.substance).cuda()

    # support different learning rate according to CORAL paper
    # i.e. 10 times learning rate for the last two fc layers.
    optimizer = torch.optim.SGD([
        {'params': model.sharedNet.parameters()},
        {'params': model.regression_predictor.parameters(), 'lr': 10*LEARNING_RATE},
    ], lr=LEARNING_RATE, momentum=MOMENTUM)

    model = model.cuda()

    # if args.load is not None:
    #     utils.load_net(model, args.load)
    # else:
    #     load_pretrained(model.sharedNet)

    best_whole_rmse = 1000

    for epoch in range(0, args.epochs):
        _lambda = (epoch+1)/args.epochs * 100        
        train(model, optimizer, epoch+1, _lambda)

        validation(model, source_val_loader, target_val_loader)

        whole_mse, whole_rmse = inference(model, target_whole_loader)

        if whole_rmse < best_whole_rmse:
            best_whole_rmse = whole_rmse

            save_dir = args.save_dir + f'coral_{args.backbone}/'
            os.makedirs(save_dir, exist_ok=True)
            save_path = save_dir + f'{args.backbone}.pth'

            torch.save(model.state_dict(), save_path)
            print(f'Model saved to {save_path} at the end of epoch {epoch}, with MSE of {whole_mse} and RMSE of {whole_rmse}.')

