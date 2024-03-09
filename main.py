import numpy as np
import random
import torch
import argparse
import time
import matplotlib.pyplot as plt

from datetime import datetime
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from utils.train import *
from utils.loader import *
from utils.utility import *
from utils.model import *
from load import *

### Make argument parser(hyper-parameters)
def get_args():
    parser = argparse.ArgumentParser()
    ### Data
    parser.add_argument('--data', default='adni', help='Type of dataset')
    parser.add_argument('--data_path', default='./data/adni_2022')
    parser.add_argument('--adjacency_path', default='/matrices2326')
    ### Condition
    parser.add_argument('--seed_num', type=int, default=1, help='Number of random seed')
    parser.add_argument('--device_num', type=int, default=7, help='Which gpu to use')
    parser.add_argument('--features', type=int, default=7, help='Features to use')
    parser.add_argument('--labels', type=int, default=6, help="Labels to use")
    parser.add_argument('--model', type=str, default='ours', help='Models to use')
    parser.add_argument('--layer_num', type=int, default=2, help='Number of layers')
    ### Experiment
    parser.add_argument('--split_num', type=int, default=5, help="Number of splits for k-fold")
    parser.add_argument('--batch_size', type=int, default=32, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs')
    parser.add_argument('--hidden_units', type=int, default=16, help='Number of hidden units')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learing rate')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='L2 loss on parameters')
    ### Experiment for "GAT"
    parser.add_argument('--num_head_attentions', type=int, default=16, help='Number of head attentions')
    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha for the leaky relu')
    ### Experiment for "Exact"
    parser.add_argument('--use_t_local', type=int, default=1, help='Whether t is local or global (0:global / 1:local)')
    parser.add_argument('--t_alpha', type=float, default=1, help='t lambda of loss function')
    parser.add_argument('--t_lr', type=float, default=10, help='t learning rate')
    parser.add_argument('--t_loss_threshold', type=float, default=0.01, help='t loss threshold')
    parser.add_argument('--t_threshold', type=float, default=0.1, help='t threshold')
    ### Experiment for "LSAP"
    parser.add_argument('--polynomial', type=int, default=2, help='Which polynomial is used (0:Chebyshev, 1:Hermite, 2:Laguerre)')
    parser.add_argument('--m_chebyshev', type=int, default=20, help='Expansion degree of Chebyshev')
    parser.add_argument('--m_hermite', type=int, default=30, help='Expansion degree of Hermite')
    parser.add_argument('--m_laguerre', type=int, default=20, help='Expansion degree of Laguerre')
    ### Etc
    parser.add_argument('--ratio', type=int, default=1, help='Contrastive learning ratio (pos : neg)')
    parser.add_argument('--only_pos', action="store_true", help='Whethre only using positive pairs')
    parser.add_argument('--LOAD', action="store_true", help='Load the data')
    parser.add_argument('--TEST', action="store_true", help='Load the data')
    args = parser.parse_args()
    
    return args

### Control the randomness of all experiments
def set_randomness(seed_num):
    torch.manual_seed(seed_num) # Pytorch randomness
    np.random.seed(seed_num) # Numpy randomness
    random.seed(seed_num) # Python randomness
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_num) # Current GPU randomness
        torch.cuda.manual_seed_all(seed_num) # Multi GPU randomness

### Main function
def main():
    args = get_args()
    #wandb.config.update(args)
    set_randomness(args.seed_num)
    device = torch.device('cuda:' + str(args.device_num) if torch.cuda.is_available() else 'cpu')
    
    DATA = args.data
    SPLIT = args.split_num
    MODEL = args.model
    TEST = args.TEST
    
    used_features, A, X, y, eigenvalues, eigenvectors, laplacians = load_dataset(args)
    
    print(X.shape)
    
    "K-fold cross validation"
    stratified_train_test_split = StratifiedKFold(n_splits=SPLIT)

    avl, ava, avac, avpr, avsp, avse, avf1s, ts = list([] for _ in range(8))
    idx_pairs = []
    for train_idx, test_idx in stratified_train_test_split.split(A, y):
        idx_train = torch.LongTensor(train_idx)
        idx_test = torch.LongTensor(test_idx)
        idx_pairs.append((idx_train, idx_test))

    "Utilize GPUs for computation"
    if torch.cuda.is_available() and MODEL != 'svm':
        A = A.to(device) # Shape: (# subjects, # ROI feature, # ROI X)
        X = X.to(device) # Shape: (# subjects, # ROI X, # used X)
        y = y.to(device) # Shape: (# subjects)
    
        eigenvalues = eigenvalues.to(device) # Shape: (# subjects, # ROI feature)
        eigenvectors = eigenvectors.to(device) # Shape: (# subject, # ROI_feature, # ROI_feature)
        laplacians = laplacians.to(device)
    
    ### Compute polynomial
    if MODEL == 'lsap':
        if args.polynomial == 0: # Chebyshev
            b = eigenvalues[:,eigenvalues.shape[1]-1] 
            P_n = compute_Tn(laplacians, args.m_chebyshev, b, device)
        elif args.polynomial == 1: # Hermite
            P_n = compute_Hn(laplacians, args.m_hermite, device)
        elif args.polynomial == 2: # Laguerre
            P_n = compute_Ln(laplacians, args.m_laguerre, device)
    else:
        P_n = None
    # pdb.set_trace()
    # P_n = None
    
    if MODEL != 'svm':
        num_ROI_features = X.shape[1]
        num_used_features = X.shape[2]
    else:
        num_ROI_features = None
        num_used_features = None

    for i, idx_pair in enumerate(idx_pairs):
        print("\n")
        print(f"=============================== Fold {i+1} ===============================")

        "Build data loader"
        data_loader_train, data_loader_test = build_data_loader(args, idx_pair, A, X, y, eigenvalues, eigenvectors, laplacians, P_n)

        "Select the model to use"
        model = select_model(args, num_ROI_features, num_used_features, A, y, device)
        optimizer = select_optimizer(args, model)
        trainer = select_trainer(args, device, model, optimizer, data_loader_train, data_loader_test, A)

        "Train and test"
        if TEST == False:
            trainer.train(i+1)

        losses, accuracies, cf_accuracies, cf_precisions, cf_specificities, cf_sensitivities, cf_f1score, t = trainer.test(i+1)
        
        avl.append(losses)
        ava.append(accuracies)
        avac.append(cf_accuracies)
        avpr.append(cf_precisions)
        avsp.append(cf_specificities)
        avse.append(cf_sensitivities)
        avf1s.append(cf_f1score)
        ts.append(t)
    
    class_info = y.tolist()
    cnt = Counter(class_info)

    ### Show parameters
    print("------------- Parameters -------------")
    print('\033[93m' + f"featrues: {args.features}" + '\033[0m')
    print('\033[93m' + f"labels: {args.labels}" + '\033[0m')
    print('\033[93m' + f"model: {args.model}" + '\033[0m')
    print('\033[93m' + f"batch_size: {args.batch_size}" + '\033[0m')
    print('\033[93m' + f"hidden_units: {args.hidden_units}" + '\033[0m')
    print('\033[93m' + f"lr: {args.lr}" + '\033[0m')
    print('\033[93m' + f"use_t_local: {args.use_t_local}" + '\033[0m')
    print('\033[93m' + f"t_alpha: {args.t_alpha}" + '\033[0m')
    print('\033[93m' + f"t_lr: {args.t_lr}" + '\033[0m')
    print('\033[93m' + f"t_loss_threshold: {args.t_loss_threshold}" + '\033[0m')
    print('\033[93m' + f"t_threshold: {args.t_threshold}" + '\033[0m')
    ### Show results
    print("--------------- Result ---------------")
    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    if DATA == 'adni':
        print(f"Used X:        {used_features}")
    print(f"Label distribution:   {cnt}")
    print(f"{args.split_num}-Fold test loss:     {avl}")
    print(f"{args.split_num}-Fold test accuracy: {ava}")
    print("---------- Confusion Matrix ----------")
    print(f"{args.split_num}-Fold accuracy:      {avac}")
    print(f"{args.split_num}-Fold precision:     {avpr}")
    print(f"{args.split_num}-Fold specificity:   {avsp}")
    print(f"{args.split_num}=Fold sensitivity:   {avse}")
    print(f"{args.split_num}=Fold f1 score:      {avf1s}")
    print("-------------- Mean, Std --------------")
    print('\033[94m' + f"Mean: {np.mean(avac):.3f} {np.mean(avpr):.3f} {np.mean(avsp):.3f} {np.mean(avse):.3f}" + '\033[0m')
    print('\033[94m' + f"Std:  {np.std(avac):.3f} {np.std(avpr):.3f} {np.std(avsp):.3f} {np.std(avse):.3f}" + '\033[0m')

if __name__ == '__main__':
    start_time = time.time()
    
    main()

    process_time = time.time() - start_time
    hour = int(process_time // 3600)
    minute = int((process_time - hour * 3600) // 60)
    second = int(process_time % 60)
    print(f"\nTime: {hour}:{minute}:{second}")
    now = datetime.now()
    print(f"▶ {now.year}-{now.month}-{now.day} {now.hour+9}:{now.minute}:{now.second} ◀")