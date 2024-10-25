import torch
import numpy as np
from sklearn.metrics import mean_squared_error
import torch.nn as nn
from model import DARE
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



# def inference(model, target_test_loader):
#     with torch.no_grad():
#         model.cuda()
#         model.eval()

#         target_mse = 0
#         num_batches = 0

#         target_pred = []
#         target_gt = []

#         regression_criterion = nn.SmoothL1Loss().cuda()
#         target_regression_loss = 0

#         for target_data in target_test_loader:

#             # Process source and target data
#             target_x, target_y_reg, target_domain = target_data
#             target_x, target_y_reg, target_domain = target_x.cuda(), target_y_reg.cuda(), target_domain.cuda()

#             # Feature Extraction
#             target_feature = model.feature_extractor(target_x)

#             # Regression Predictions
#             target_pred_batch = model.regression_predictor(target_feature)
#             target_pred += torch.squeeze(target_pred_batch)
#             target_gt += target_y_reg
            
#             # Compute loss for debuging
#             target_regression_loss += regression_criterion(target_pred_batch.view(-1), target_y_reg).item()
#             num_batches += 1

#     target_pred = [tensor.to('cpu').item() for tensor in target_pred]
#     target_gt = [tensor.to('cpu').item() for tensor in target_gt]
#     target_mse = mean_squared_error(target_pred, target_gt)
#     target_rmse = np.sqrt(target_mse)

#     print(f'[The whole target set - {len(target_gt)} samples]\tTarget Regression Loss: {target_regression_loss/num_batches:.4f}\tMSE: {target_mse}\tRMSE: {target_rmse}\n')

#     save_predictions_to_excel(target_pred, target_gt, "/mnt/beegfs/home/liu15/la-ood-tl-main/DANN/xlsx_files/source_glucose_predictions.xlsx")


#     return target_rmse


def Regression_test(model, target_test_loader):
    model.eval()
    target_pred = []
    target_gt = []

    regression_criterion = nn.MSELoss().cuda()
    target_regression_loss = 0
    num_batches = 0
    with torch.no_grad():
        for target_data in target_test_loader:
            # Process source and target data
            target_x, target_y_reg, target_domain = target_data
            target_x, target_y_reg, target_domain = target_x.cuda(), target_y_reg.cuda(), target_domain.cuda()
    
            target_pred_batch = model(target_x)

            # Regression Predictions
            target_pred += torch.squeeze(target_pred_batch, dim=1)
            target_gt += target_y_reg
            
            # Compute loss for debuging
            target_regression_loss += regression_criterion(target_pred_batch.view(-1), target_y_reg).item()
            num_batches += 1

        target_pred = [tensor.to('cpu').item() for tensor in target_pred]
        target_gt = [tensor.to('cpu').item() for tensor in target_gt]
        target_mse = mean_squared_error(target_pred, target_gt)
        target_rmse = np.sqrt(target_mse)

    print(f'[The target set - {len(target_gt)} samples]\tTarget MSE Loss: {target_regression_loss/num_batches:.4f}\tMSE: {target_mse}\tRMSE: {target_rmse}')

    save_predictions_to_excel(target_pred, target_gt, "/mnt/beegfs/home/liu15/la-ood-tl-main/DARE-GRAM/DARE-GRAM_xlsx/target_lacticacid_predictions.xlsx")


    return target_mse, target_rmse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='PyTorch DARE-GRAM experiment')
    parser.add_argument('--backbone', type=str, default='resnet101', help="model backbone")
    parser.add_argument('--substance', type=str, default='lacticacid', choices=['glucose', 'lacticacid'])

    args = parser.parse_args()

    model = DARE(backbone=args.backbone, substance=args.substance).cuda()

    weight_path = '' # Please place your trained model weights here
    model.load_state_dict(torch.load(weight_path))

    target_whole_loader = dataset.target_lacticacid_whole_loader
    Regression_test(model.predict_layer, target_whole_loader)
