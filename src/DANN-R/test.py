import torch
import numpy as np
from sklearn.metrics import mean_squared_error
import torch.nn as nn
from model import DANN
import dataset
import pandas as pd
import argparse


def save_predictions_to_excel(predictions, ground_truth, filename):
    # Create a DataFrame with predictions and ground truth
    df = pd.DataFrame({
        'Prediction': predictions,
        'Ground Truth': ground_truth
    })

    # Save DataFrame to Excel file
    df.to_excel(filename, index=False)
    print(f'Saved predictions and ground truth to {filename}')


def tester(model, source_test_loader, target_test_loader, training_mode):
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
        if training_mode == 'DANN':
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

    print(f"[Source Regression] - Regression Loss: {source_regression_loss/num_batches:.4f}\tMSE: {source_mse:.3f}\tRMSE: {np.sqrt(source_mse):.3f}\n"
      f"[Target Regression] - Regression Loss: {target_regression_loss/num_batches:.4f}\tMSE: {target_mse:.3f}\tRMSE: {np.sqrt(target_mse):.3f}")
    
    if training_mode == "DAAN":
        print(f"[Domain Classification] - Correct: {domain_correct}\tTotal: {domain_total}\tAccuracy: {100.*domain_correct/domain_total:.3f}%")


def inference(model, target_test_loader):
    model.eval()

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

    print(f'[The whole target set - {len(target_gt)} samples]\tTarget Regression Loss: {target_regression_loss/num_batches:.4f}\tMSE: {target_mse}\tRMSE: {target_rmse}\n')

    save_predictions_to_excel(target_pred, target_gt, "/mnt/beegfs/home/liu15/la-ood-tl-main/DANN/DANN_xlsx/target_lacticacid_predictions.xlsx")


    return target_rmse


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='PyTorch DANN experiment')
    parser.add_argument('--backbone', type=str, default='resnet101', help="model backbone")
    parser.add_argument('--substance', type=str, default='lacticacid', choices=['glucose', 'lacticacid'])

    args = parser.parse_args()


    model = DANN(backbone=args.backbone, substance=args.substance).cuda()

    weight_path = '/resnet101_6.pth'
    model.load_state_dict(torch.load(weight_path))

    target_whole_loader = dataset.target_lacticacid_whole_loader
    inference(model, target_whole_loader)
