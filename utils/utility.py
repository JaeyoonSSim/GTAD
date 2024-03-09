import torch
import numpy as np

### Select the feature files and labels
def set_classification_conditions(args):
    "Feature selection"
    features_list_all = {0: ['DT_thickness.xlsx', 'Amyloid_SUVR.xlsx', 'FDG_SUVR.xlsx'], # Degree
                         1: ['DT_thickness.xlsx', 'Amyloid_SUVR.xlsx', 'FDG_SUVR.xlsx'], # Cortical Thickness
                         2: ['DT_thickness.xlsx', 'Amyloid_SUVR.xlsx', 'FDG_SUVR.xlsx'], # Amyloid
                         3: ['DT_thickness.xlsx', 'Amyloid_SUVR.xlsx', 'FDG_SUVR.xlsx'], # FDG
                         
                         4: ['DT_thickness.xlsx', 'Amyloid_SUVR.xlsx', 'FDG_SUVR.xlsx'], # Cortical Thickness & Amyloid
                         5: ['DT_thickness.xlsx', 'Amyloid_SUVR.xlsx', 'FDG_SUVR.xlsx'], # Cortical Thickness & FDG
                         6: ['DT_thickness.xlsx', 'Amyloid_SUVR.xlsx', 'FDG_SUVR.xlsx'], # Amyloid & FDG
                         
                         7: ['DT_thickness.xlsx', 'Amyloid_SUVR.xlsx', 'FDG_SUVR.xlsx'] # Cortical Thickness & Amyloid & FDG
                         }
    "Label selection"
    labels_dict_all = {0: {'CN': 0, 'SMC': 1, 'EMCI': 2, 'LMCI': 3, 'AD': 4}, # 5-way ( CN vs SMC vs EMCI vs LMCI vs AD )
                       1: {'CN': 0, 'SMC': 0, 'EMCI': 1, 'LMCI': 1, 'AD': 2}, # 3-way ( CN + SMC vs EMCI + LMCI vs AD )
                       
                       2: {'CN': 0, 'SMC': 0, 'EMCI': 1, 'LMCI': 1}, # Pre vs. MCI
                       3: {'CN': 0, 'SMC': 0, 'AD': 1}, # Pre vs. AD
                       
                       4:{'EMCI': 0, 'LMCI': 0, 'AD': 1}, # MCI vs. AD
                       5:{'EMCI': 0, 'LMCI': 1},
                       
                       6:{'CN': 0, 'SMC': 1, 'EMCI':2}
                       }
    
    features_list = features_list_all[args.features] # Select feature files
    labels_dict = labels_dict_all[args.labels] # Select classification labels
    
    return features_list, labels_dict


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

### Heat kernel filtering
def hk_filtering(features, eigenvalues, eigenvectors, t):
    """
    features : (# of samples, # of ROIs, # of features)
    eigenvalue : (# of samples, # of ROIs)
    eigenvector : (# of samples, # of ROIs, # of ROIs)
    t : (# of ROIs) or (1)
    """
    filtered_features = []

    num_samples = features.shape[0]
    
    for i in range(num_samples):
        f = features[i] # (160, 1)
        eigenvalue = eigenvalues[i] # (160)
        eigenvector = eigenvectors[i] # (160, 160)
        
        f_hat = torch.matmul(eigenvector.T, f) # (160, 1)
        
        exponent = torch.mul(-eigenvalue, t).reshape(-1, 1) # (160, 1)

        g = torch.exp(exponent) # (160, 1)
        
        g_f_hat = torch.mul(g, f_hat) # (160. 1)
        
        reconstructed_f = torch.matmul(eigenvector, g_f_hat) # (160, 1)
        filtered_features.append(reconstructed_f)

    filtered_features = torch.stack(filtered_features)

    return filtered_features