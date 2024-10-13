import torch
import numpy as np

### Normalize the feature matrix
def normalize_feature(X):
    X = X.T
    normalized_X = (X - X.mean(1).reshape(-1, 1)) / (torch.max(X, dim=1).values - torch.min(X, dim=1).values).reshape(-1, 1)
    normalized_X = normalized_X.T
    
    return normalized_X

### Normalize the adjacency matrix
def normalize_adjacency(A):
    rowsum = np.array(A.sum(1)) # Number of connected edges 
    r_inv = np.power(rowsum, -1).flatten() # Inverse of number of connected edges
    r_inv[np.isinf(r_inv)] = 0. # Check whether each element is infinity and map to 0
    r_mat_inv = np.diag(r_inv) # Make diagonal matrix 
    
    normalized_A = r_mat_inv.dot(A) # Matrix multiplication with original adjacency matrix and diagonal matrix we made
    normalized_A = torch.FloatTensor(normalized_A)

    return normalized_A

def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    
    return labels_onehot