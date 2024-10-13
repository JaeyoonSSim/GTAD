import numpy as np
import random
import torch
import argparse
import time

from sklearn.model_selection import StratifiedKFold
from datetime import datetime
from collections import Counter
from utils.train import *
from utils.loader import *
from utils.utility import *
from utils.model import *

### Make argument parser(hyper-parameters)
def get_args():
    parser = argparse.ArgumentParser()
    ### Data
    parser.add_argument('--data', default='adni', help='Type of dataset')
    ### Condition
    parser.add_argument('--seed', type=int, default=1, help='Number of random seed')
    parser.add_argument('--device', type=int, default=7, help='Which gpu to use')
    parser.add_argument('--feature', type=int, default=4, help='Features to use')
    parser.add_argument('--label', type=int, default=2, help="Labels to use")
    parser.add_argument('--model', type=str, default='gtad', help='Models to use')
    parser.add_argument('--layer', type=int, default=2, help='Number of layers')
    ### Experiment
    parser.add_argument('--split', type=int, default=5, help="Number of splits for k-fold")
    parser.add_argument('--batch', type=int, default=32, help='Input batch size for training')
    parser.add_argument('--epoch', type=int, default=2000, help='Number of epochs')
    parser.add_argument('--hid', type=int, default=16, help='Number of hidden units')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learing rate')
    parser.add_argument('--dr', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--wd', type=float, default=5e-4, help='L2 loss on parameters')
    ### Etc
    parser.add_argument('--TEST', action="store_true", help='Load the data')
    args = parser.parse_args()
    
    return args

### Control the randomness of all experiments
def set_randomness(seed):
    torch.manual_seed(seed) # Pytorch randomness
    np.random.seed(seed) # Numpy randomness
    random.seed(seed) # Python randomness
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # Current GPU randomness
        torch.cuda.manual_seed_all(seed) # Multi GPU randomness

### Main function
def main():
    args = get_args()
    set_randomness(args.seed)
    device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')
    
    A, X, Y, L, EIGVAL, EIGVEC = load_dataset(args)
    print(X.shape)
    
    "K-fold cross validation"
    stratified_train_test_split = StratifiedKFold(n_splits=args.split)

    avl, ava, avac, avpr, avsp, avse, avf1s, ts = list([] for _ in range(8))
    idx_pairs = []
    for train_idx, test_idx in stratified_train_test_split.split(A, Y):
        idx_train = torch.LongTensor(train_idx)
        idx_test = torch.LongTensor(test_idx)
        idx_pairs.append((idx_train, idx_test))

    "Utilize GPUs for computation"
    if torch.cuda.is_available():
        A = A.to(device) # (sample, roi, roi)
        X = X.to(device) # (sample, roi, feat)
        Y = Y.to(device) # (sample)
        L = L.to(device) # (roi, roi)
        EIGVAL = EIGVAL.to(device) # (sample, roi)
        EIGVEC = EIGVEC.to(device) # (sample, roi, roi)
    
    n_roi = X.shape[1]
    n_modal = X.shape[2]

    for i, idx_pair in enumerate(idx_pairs):
        print(f"=============================== Fold {i+1} ===============================")

        "Build data loader"
        data_loader_train, data_loader_test = build_data_loader(idx_pair, A, X, Y, L, EIGVAL, EIGVEC)

        "Select the model to use"
        model = select_model(args, n_roi, n_modal, Y)
        optimizer = select_optimizer(args, model)
        trainer = select_trainer(args, device, model, optimizer, data_loader_train, data_loader_test)

        "Train and test"
        if args.TEST == False:
            trainer.train(i+1)

        losses, acc, cf_acc, cf_pre, cf_spe, cf_sen, cf_f1s, t  = trainer.test(i+1)
        
        avl.append(losses)
        ava.append(acc)
        avac.append(cf_acc)
        avpr.append(cf_pre)
        avsp.append(cf_spe)
        avse.append(cf_sen)
        avf1s.append(cf_f1s)
        ts.append(t)
    
    class_info = Y.tolist()
    cnt = Counter(class_info)

    ### Show parameters
    print("------------- Parameters -------------")
    print('\033[93m' + f"featrue: {args.feature}" + '\033[0m')
    print('\033[93m' + f"label: {args.label}" + '\033[0m')
    print('\033[93m' + f"model: {args.model}" + '\033[0m')
    print('\033[93m' + f"batch: {args.batch}" + '\033[0m')
    print('\033[93m' + f"epoch: {args.epoch}" + '\033[0m')
    print('\033[93m' + f"hid: {args.hid}" + '\033[0m')
    print('\033[93m' + f"lr: {args.lr}" + '\033[0m')
    print('\033[93m' + f"wd: {args.wd}" + '\033[0m')
    ### Show results
    print("--------------- Result ---------------")
    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    print(f"Label distribution:   {cnt}")
    print(f"{args.split}-Fold test loss:     {avl}")
    print(f"{args.split}-Fold test accuracy: {ava}")
    print("---------- Confusion Matrix ----------")
    print(f"{args.split}-Fold accuracy:      {avac}")
    print(f"{args.split}-Fold precision:     {avpr}")
    print(f"{args.split}-Fold specificity:   {avsp}")
    print(f"{args.split}=Fold sensitivity:   {avse}")
    print(f"{args.split}=Fold f1 score:      {avf1s}")
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