import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from sklearn.metrics import mean_squared_error
from model import DARE
import random
from dataset import SugarDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser(description='PyTorch DARE-GRAM experiment')
parser.add_argument('--backbone', type=str, default='transformer', help="model backbone")
parser.add_argument('--substance', type=str, default='glucose', choices=['glucose', 'lacticacid'])

parser.add_argument('--dataset_dir', type=str, default='', help="source dataset file")
parser.add_argument('--save_dir', type=str, default='', help="save model firectory")

parser.add_argument('--gpu_id', type=str, nargs='?', default='3', help="device id to run")

parser.add_argument('--lr', type=float, default=0.01, help='init learning rate for fine-tune')
parser.add_argument('--gamma', type=float, default=0.0001, help='learning rate decay')
parser.add_argument('--tradeoff_angle', type=float, default=0.05, help='tradeoff for angle alignment')
parser.add_argument('--tradeoff_scale', type=float, default=0.01, help='tradeoff for scale alignment')
parser.add_argument('--treshold', type=float, default=0.9, help='treshold for the pseudo inverse')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--num_iter', type=int, default=8000, help='iteration to train')
parser.add_argument('--val_interval', type=int, default=100, help='iteration to validate')
parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
parser.add_argument('--num_runs', type=int, default=10, help='num of experiment runs')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

args.source_path = args.dataset_dir + f'ss_{args.substance}.xlsx'
args.target_path = args.dataset_dir + f'cs_{args.substance}.xlsx'
args.save_dir = args.save_dir + f'runs_{args.substance}/'

print(f"Source Dataset: {args.source_path}")
print(f"Target Dataset: {args.target_path}")
print()

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"Device: {device}\n")

def Regression_test(target_test_loader, model):
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

    return target_mse, target_rmse

def DARE_GRAM_LOSS(H1, H2):    
    b,p = H1.shape

    A = torch.cat((torch.ones(b,1).cuda(), H1), 1)
    B = torch.cat((torch.ones(b,1).cuda(), H2), 1)

    cov_A = (A.t()@A)
    cov_B = (B.t()@B) 

    _,L_A,_ = torch.linalg.svd(cov_A)
    _,L_B,_ = torch.linalg.svd(cov_B)
    
    eigen_A = torch.cumsum(L_A.detach(), dim=0)/L_A.sum()
    eigen_B = torch.cumsum(L_B.detach(), dim=0)/L_B.sum()

    if(eigen_A[1]>args.treshold):
        T = eigen_A[1].detach()
    else:
        T = args.treshold
        
    index_A = torch.argwhere(eigen_A.detach()<=T)[-1]

    if(eigen_B[1]>args.treshold):
        T = eigen_B[1].detach()
    else:
        T = args.treshold

    index_B = torch.argwhere(eigen_B.detach()<=T)[-1]
    
    k = max(index_A, index_B)[0]

    A = torch.linalg.pinv(cov_A ,rtol = (L_A[k]/L_A[0]).detach())
    B = torch.linalg.pinv(cov_B ,rtol = (L_B[k]/L_B[0]).detach())
    
    cos_sim = nn.CosineSimilarity(dim=0,eps=1e-6)
    cos = torch.dist(torch.ones((p+1)).cuda(),(cos_sim(A,B)),p=1)/(p+1)
    
    return args.tradeoff_angle*(cos) + args.tradeoff_scale*torch.dist((L_A[:k]),(L_B[:k]))/k

def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma, power, init_lr=0.001, weight_decay=0.0005):
    lr = init_lr * (1 + gamma * iter_num) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        param_group['weight_decay'] = weight_decay * 2
        i += 1
    return optimizer


def main():

    MSE = []
    RMSE = []

    for run in tqdm(range(args.num_runs), desc="Total Runs", unit="run"):

        # Set a different random seed for each run
        random_seed = 2024
        torch.manual_seed(random_seed)
        random.seed(random_seed)

        print(f"[{run}/{args.num_runs} runs] Loading Dataset...")
        source_glucose_dataset = SugarDataset(data_path=args.source_path, source_domain=True, test_ratio=0.15)
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

        print(f"[{run}/{args.num_runs} runs] Creating DARE_GRAM model with {args.backbone} backbone on {args.substance} prediction... ")
        Model_R = DARE(backbone=args.backbone, substance=args.substance).cuda()


        criterion = {"regressor": nn.MSELoss()}
        # optimizer_dict = [{"params": filter(lambda p: p.requires_grad, Model_R.model_fc.parameters()), "lr": 0.01},
        #                   {"params": filter(lambda p: p.requires_grad, Model_R.classifier_layer.parameters()), "lr": 0.1}]

        optimizer = torch.optim.Adam(Model_R.parameters(), lr=args.lr)  # Reduced learning rate
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        param_lr = []
        for param_group in optimizer.param_groups:
            param_lr.append(param_group["lr"])


        len_source = len(source_train_loader) - 1
        len_target = len(target_train_loader) - 1

        iter_source = iter(source_train_loader)
        iter_target = iter(target_train_loader)


        train_regression_loss = train_dare_gram_loss = train_total_loss =  0.0
        best_val_rmse = 1000
        best_whole_rmse = 1000
        best_whole_mse = 10000

        print(f"[{run}/{args.num_runs} runs] Start Training...")
        for iter_num in range(1, args.num_iter + 1):

            Model_R.train()
            optimizer.zero_grad()
            optimizer = inv_lr_scheduler(param_lr, optimizer, iter_num, init_lr=args.lr, gamma=args.gamma, power=0.75, weight_decay=0.05)

            if iter_num % len_source == 0:
                iter_source = iter(source_train_loader)
            if iter_num % len_target == 0:
                iter_target = iter(target_train_loader)

            source_data = iter_source.next()
            target_data = iter_target.next()
            
            source_x, source_y_reg, source_domain = source_data
            target_x, target_y_reg, target_domain = target_data

            source_x, source_y_reg, source_domain = source_x.cuda(), source_y_reg.cuda(), source_domain.cuda() 
            target_x, target_y_reg, target_domain = target_x.cuda(), target_y_reg.cuda(), target_domain.cuda() 
            

            source_y_reg = source_y_reg.unsqueeze(1)

            outC_s, feature_s = Model_R(source_x)
            outC_t, feature_t = Model_R(target_x)


            regression_loss = criterion["regressor"](outC_s, source_y_reg)
            dare_gram_loss = DARE_GRAM_LOSS(feature_s,feature_t)

            total_loss = regression_loss + dare_gram_loss * Model_R.alpha_domain

            total_loss.backward()
            optimizer.step()

            train_regression_loss += regression_loss.item()
            train_dare_gram_loss += dare_gram_loss.item()
            train_total_loss += total_loss.item()

            if iter_num % args.val_interval == 0:
                print("Iter {:05d}/{:05d}, Average MSE Loss: {:.5f}; Average DARE-GRAM Loss: {:.5f}; Average Training Loss: {:.5f}".format(
                    iter_num, args.num_iter, train_regression_loss / float(args.val_interval), train_dare_gram_loss / float(args.val_interval), train_total_loss / float(args.val_interval)))
                train_regression_loss = train_dare_gram_loss = train_total_loss =  0.0
                
                Model_R.eval()
                val_mse, val_rmse = Regression_test(target_val_loader, Model_R.predict_layer)

                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    whole_mse, whole_rmse = Regression_test(target_whole_loader, Model_R.predict_layer)

                    if whole_rmse < best_whole_rmse:
                        best_whole_rmse = whole_rmse
                        best_whole_mse = whole_mse

                        save_dir = args.save_dir + f'dare_{args.backbone}/'
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = save_dir + f'{args.backbone}_{run}.pth'

                        torch.save(Model_R.state_dict(), save_path)
                        print(f'[{run}/{args.num_runs}] Model saved to {save_path} at the end of iter {iter_num}, with MSE of {best_whole_mse} and RMSE of {best_whole_rmse}.\n')

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