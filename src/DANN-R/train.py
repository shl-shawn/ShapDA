import os
import torch
from dataset import SugarDataset
from model import DANN
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import argparse
import numpy as np
import random
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch DANN experiment')
parser.add_argument('--backbone', type=str, default='resnet101', help="model backbone")
parser.add_argument('--substance', type=str, default='lacticacid', choices=['glucose', 'lacticacid'])

parser.add_argument('--dataset_dir', type=str, default=f'', help="source dataset file")
parser.add_argument('--save_dir', type=str, default='', help="save model firectory")

parser.add_argument('--gpu_id', type=str, nargs='?', default='1', help="device id to run")
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
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

def inference(model, target_test_loader):
    with torch.no_grad():
        model.cuda()
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
            target_feature = model.feature_extractor(target_x)

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
    model.cuda()
    model.eval()

    domain_correct = 0
    domain_total = 0

    source_mse = 0
    target_mse = 0

    source_pred = []
    target_pred = []

    source_gt = []
    target_gt = []

    num_batches = 0

    source_regression_loss = 0
    target_regression_loss = 0

    criterion_reg = nn.SmoothL1Loss().cuda()

    for batch_idx, (source_data, target_data) in enumerate(zip(source_test_loader, target_test_loader)):
        p = float(batch_idx) / len(source_test_loader)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # 1. Process source and target data with same batch size
        real_batch_size = min(source_data[0].size(0), target_data[0].size(0))
        source_data = [data[:real_batch_size] for data in source_data]
        target_data = [data[:real_batch_size] for data in target_data]

        source_x, source_y_reg, source_domain = source_data
        target_x, target_y_reg, target_domain = target_data 

        source_x, source_y_reg, source_domain = source_x.cuda(), source_y_reg.cuda(), source_domain.cuda() 
        target_x, target_y_reg, target_domain = target_x.cuda(), target_y_reg.cuda(), target_domain.cuda() 

        # 2. Feature Extraction
        source_feature = model.feature_extractor(source_x)
        target_feature = model.feature_extractor(target_x)

        # 3. Regression Predictions
        source_pred_batch = model.regression_predictor(source_feature)
        target_pred_batch = model.regression_predictor(target_feature)

        source_pred += torch.squeeze(source_pred_batch)
        target_pred += torch.squeeze(target_pred_batch)
        source_gt += source_y_reg
        target_gt += target_y_reg

        # 4. Domain classification
        # 4.1 Process combined images
        combined_image = torch.cat((source_x, target_x), 0)
        domain_labels = torch.cat((torch.zeros(source_domain.size(0), dtype=torch.long),
                                    torch.ones(target_domain.size(0), dtype=torch.long)), 0).cuda()

        # 4.2 Compute domain predictions accuracy
        domain_pred = model(combined_image, alpha=alpha, task='domain').data.max(1, keepdim=True)[1]
        domain_correct += domain_pred.eq(domain_labels.data.view_as(domain_pred)).sum().item()
        domain_total += len(domain_labels)

        
        # 5. Compute loss
        source_regression_loss += criterion_reg(source_pred_batch.view(-1), source_y_reg).item()
        target_regression_loss += criterion_reg(target_pred_batch.view(-1), target_y_reg).item()
        num_batches += 1

    # 6. Compute error and summary
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



def main():

    MSE = []
    RMSE = []

    for run in tqdm(range(args.num_runs), desc="Total Runs", unit="run"):

        # Set a different random seed for each run
        random_seed = 2024 #andom.randint(0, 10000)
        torch.manual_seed(random_seed)
        random.seed(random_seed)

        print(f"[{run}/{args.num_runs} runs] Loading Dataset...")
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


        print(f"[{run}/{args.num_runs} runs] Creating DANN model with {args.backbone} backbone... ")
        model = DANN(backbone=args.backbone, substance=args.substance).cuda()

        criterion_reg = nn.SmoothL1Loss().cuda()
        criterion_domain = nn.CrossEntropyLoss().cuda()
        
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

        best_val_rmse = 1000
        best_whole_rmse = 1000
        best_whole_mse = 10000

        print(f"[{run}/{args.num_runs} runs] Start Training...")
        for epoch in range(args.epochs):

            model.train()

            start_steps = epoch * len(source_train_loader)
            total_steps = args.epochs * len(target_train_loader)
            total_loss_epoch = 0
            regression_loss_total = 0
            domain_loss_total = 0
            num_batches = 0

            for batch_idx, (source_data, target_data) in enumerate(zip(source_train_loader, target_train_loader)):

                source_x, source_y_reg, source_domain = source_data
                target_x, target_y_reg, target_domain = target_data

                p = float(batch_idx + start_steps) / total_steps
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

                source_x, source_y_reg, source_domain = source_x.cuda(), source_y_reg.cuda(), source_domain.cuda() 
                target_x, target_y_reg, target_domain = target_x.cuda(), target_y_reg.cuda(), target_domain.cuda() 
                combined_x = torch.cat((source_x, target_x), 0)

                optimizer.zero_grad()

                # 1. Feature Extraction
                combined_feature = model.feature_extractor(combined_x)
                source_feature = model.feature_extractor(source_x)

                # 2. Regression loss
                reg_pred = model.regression_predictor(source_feature)
                regression_loss = criterion_reg(reg_pred.view(-1), source_y_reg)

                # 3. Domain loss
                domain_pred = model.domain_classifier(combined_feature, alpha)

                domain_source_labels = torch.zeros(source_domain.shape[0]).type(torch.LongTensor)
                domain_target_labels = torch.ones(target_domain.shape[0]).type(torch.LongTensor)
                domain_combined_label = torch.cat((domain_source_labels, domain_target_labels), 0).cuda()
                domain_loss = criterion_domain(domain_pred, domain_combined_label)

                # 4. Total loss
                total_loss = regression_loss + model.alpha_domain * domain_loss
                total_loss.backward()
                optimizer.step()

                regression_loss_total += regression_loss.item()
                domain_loss_total += domain_loss.item()
                total_loss_epoch += total_loss.item()
                num_batches += 1

            scheduler.step()    
            print(f'[Epoch {epoch}]\tTotal Loss: {total_loss_epoch/num_batches:.4f}\tRegression Loss: {regression_loss_total/num_batches:.4f}\tDomain Loss: {domain_loss_total/num_batches:.4f}')

            # Evaluting on target validation set to avoid overfitting
            val_rmse = validation(model, source_val_loader, target_val_loader)


            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse

                # Predict on target whole set, removing domain discriminator, only encoder + regression branch
                whole_mse, whole_rmse = inference(model, target_whole_loader)
                if whole_rmse < best_whole_rmse:
                    best_whole_rmse = whole_rmse
                    best_whole_mse = whole_mse

                    save_dir = args.save_dir + f'dann_{args.backbone}/'
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = save_dir + f'{args.backbone}_{run}.pth'

                    torch.save(model.state_dict(), save_path)
                    print(f'[{run}/{args.num_runs}] Model saved to {save_path} at the end of epoch {epoch}, with MSE of {whole_mse} and RMSE of {whole_rmse}.')

        print(f"Best RMSE on the target whole dataset: {best_whole_rmse}")
        MSE.append(best_whole_mse)
        RMSE.append(best_whole_rmse)

    mean_MSE = np.mean(np.array(MSE))
    mean_RMSE = np.mean(np.array(RMSE))

    var_MSE = np.var(np.array(MSE), ddof=1)
    var_RMSE = np.var(np.array(RMSE), ddof=1)

    # Save to file
    output_file = save_dir + "results.txt"

    with open(output_file, "w") as f:
        f.write(f'MSE of {len(MSE)} runs: {MSE}\n')
        f.write(f'RMSE of {len(RMSE)} runs: {RMSE}\n\n\n')

        f.write(f"Mean of MSE: {mean_MSE}\n")
        f.write(f"Variance of MSE: {var_MSE}\n\n")
        f.write(f"Mean of RMSE: {mean_RMSE}\n")
        f.write(f"Variance of RMSE: {var_RMSE}\n")

    print(f"Results have been saved to {output_file}")

    


if __name__ == '__main__':
    main()