import pickle

from .utility import *
from torch.utils.data import TensorDataset, DataLoader

def load_dataset(args):
    data = args.data
    load_path = './data/'
    
    final_path = load_path + '/' + str(data) + '_' + str(args.feature) + '_' + str(args.label) + '.pickle'

    with open(final_path, 'rb') as dataset:
        dataset = pickle.load(dataset)
    
    _, A, X, Y, EIGVAL, EIGVEC, L = dataset
    
    return A, X, Y, L, EIGVAL, EIGVEC

def build_data_loader(idx_pair, A, X, Y, L, EIGVAL, EIGVEC):
    idx_train, idx_test = idx_pair
    
    data_train = TensorDataset(A[idx_train], X[idx_train], Y[idx_train], L[idx_train], EIGVAL[idx_train], EIGVEC[idx_train])
    data_test = TensorDataset(A[idx_test], X[idx_test], Y[idx_test], L[idx_test], EIGVAL[idx_test], EIGVEC[idx_test])
    
    dl_train = DataLoader(data_train, batch_size=idx_train.shape[0], shuffle=True) # Full-batch
    dl_test = DataLoader(data_test, batch_size=idx_test.shape[0], shuffle=False) # Full-batch
    
    return dl_train, dl_test