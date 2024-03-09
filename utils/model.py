import torch
import torch.optim as optim

from utils.utility import *
from utils.loader import *
from utils.train import *
from models.exact import *
from models.gcn import GCN
from models.gat import GAT
from models.mlp import MLP
from models.gdc import GDC
from models.adc import ADC
from models.lsap import *
from models.nodeformer import NodeFormer
from models.sgformer import SGFormer
from models.difformer import DIFFormer
from models.ours import OURS
from sklearn.svm import SVC, LinearSVC

def select_model(args, num_ROI_features, num_used_features, adjacencies, labels, device):
    MODEL = args.model
    LAYER = args.layer_num
    
    if MODEL == 'svm':
        model = SVC(kernel='linear')
    elif MODEL == 'mlp':
        model = MLP(in_feats = num_ROI_features,
                    hid_feats = args.hidden_units,
                    out_feats = torch.max(labels).item() + 1)
    elif MODEL == 'exact':
        if LAYER == 1:
            model = Exact1(adj_dim=num_ROI_features,
                        in_dim=num_used_features,
                        hid_dim=args.hidden_units,
                        out_dim=torch.max(labels).item() + 1,
                        dropout=args.dropout_rate)
        elif LAYER == 2:
            model = Exact2(adj_dim=num_ROI_features,
                        in_dim=num_used_features,
                        hid_dim=args.hidden_units,
                        out_dim=torch.max(labels).item() + 1,
                        dropout=args.dropout_rate)
        elif LAYER == 3:
            model = Exact3(adj_dim=num_ROI_features,
                        in_dim=num_used_features,
                        hid_dim=args.hidden_units,
                        out_dim=torch.max(labels).item() + 1,
                        dropout=args.dropout_rate)
        elif LAYER == 4:
            model = Exact4(adj_dim=num_ROI_features,
                        in_dim=num_used_features,
                        hid_dim=args.hidden_units,
                        out_dim=torch.max(labels).item() + 1,
                        dropout=args.dropout_rate)
    elif MODEL == 'lsap':
        if LAYER == 1:
            model = LSAP1(adj_dim=num_ROI_features,
                        in_dim=num_used_features,
                        hid_dim=args.hidden_units,
                        out_dim=torch.max(labels).item() + 1,
                        dropout=args.dropout_rate)
        elif LAYER == 2:
            model = LSAP2(adj_dim=num_ROI_features,
                        in_dim=num_used_features,
                        hid_dim=args.hidden_units,
                        out_dim=torch.max(labels).item() + 1,
                        dropout=args.dropout_rate)
        elif LAYER == 3:
            model = LSAP3(adj_dim=num_ROI_features,
                        in_dim=num_used_features,
                        hid_dim=args.hidden_units,
                        out_dim=torch.max(labels).item() + 1,
                        dropout=args.dropout_rate)
        elif LAYER == 4:
            model = LSAP4(adj_dim=num_ROI_features,
                        in_dim=num_used_features,
                        hid_dim=args.hidden_units,
                        out_dim=torch.max(labels).item() + 1,
                        dropout=args.dropout_rate)
    elif MODEL == 'gcn':
        """
        ajd_dim: # ROI features (edges) 
        in_dim: # used features (nodes)
        hid_dim: # hidden units (weights)
        out_dim: # labels (classes)
        """
        model = GCN(adj_dim=num_ROI_features,
                    in_dim=num_used_features,
                    hid_dim=args.hidden_units,
                    out_dim=torch.max(labels).item() + 1,
                    dropout=args.dropout_rate)
    elif MODEL == 'gdc':
        """
        ajd_dim: # ROI features (edges) 
        in_dim: # used features (nodes)
        hid_dim: # hidden units (weights)
        out_dim: # labels (classes)
        """
        model = GDC(adj_dim=num_ROI_features,
                    in_dim=num_used_features,
                    hid_dim=args.hidden_units,
                    out_dim=torch.max(labels).item() + 1,
                    dropout=args.dropout_rate)
    elif MODEL == 'graphheat':
        """
        ajd_dim: # ROI features (edges) 
        in_dim: # used features (nodes)
        hid_dim: # hidden units (weights)
        out_dim: # labels (classes)
        """
        model = GCN(adj_dim=num_ROI_features,
                    in_dim=num_used_features,
                    hid_dim=args.hidden_units,
                    out_dim=torch.max(labels).item() + 1,
                    dropout=args.dropout_rate)
    elif MODEL == 'gat':
        """
        nfeat: # used features (nodes)
        nhid: # hidden units (weights)
        nclass: # labels (classes)
        """
        model = GAT(nfeat=num_used_features,
                    nhid=args.hidden_units,
                    nclass=torch.max(labels).item() + 1,
                    dropout=args.dropout_rate,
                    alpha=args.alpha,
                    adj_sz=adjacencies[0].shape[0],
                    nheads=args.num_head_attentions)
    elif MODEL == 'adc':
        """
        nfeat: # used features (nodes)
        nhid: # hidden units (weights)
        nclass: # labels (classes)
        """
        model = ADC(adj_dim=num_ROI_features,
                    in_dim=num_used_features,
                    hid_dim=args.hidden_units,
                    out_dim=torch.max(labels).item() + 1,
                    dropout=args.dropout_rate)
    elif MODEL == 'nodeformer':
        model = NodeFormer(adj_channels=num_ROI_features,
                        in_channels=num_used_features,
                        hidden_channels=args.hidden_units,
                        out_channels=torch.max(labels).item()+1,
                        device = device)
    elif MODEL == 'sgformer':
        model = SGFormer(adj_channels=num_ROI_features,
                        in_channels=num_used_features,
                        hidden_channels=args.hidden_units,
                        out_channels=torch.max(labels).item()+1)
    elif MODEL == 'difformer':
        model = DIFFormer(adj_channels=num_ROI_features,
                        in_channels=num_used_features,
                        hidden_channels=args.hidden_units,
                        out_channels=torch.max(labels).item()+1)
    elif MODEL == 'ours':
        """
        ajd_dim: # ROI features (edges) 
        in_dim: # used features (nodes)
        hid_dim: # hidden units (weights)
        out_dim: # labels (classes)
        """
        model = OURS(adj_dim=num_ROI_features,
                    in_dim=num_used_features, # 2,4,Ours
                    # in_dim=1, # 1,3,5
                    hid_dim=args.hidden_units,
                    out_dim=torch.max(labels).item() + 1,
                    ratio=args.ratio,
                    dropout=args.dropout_rate)
    else:
        raise ValueError
    
    return model
        
def select_optimizer(args, model):
    if args.model == 'ours':
        scale_params = [
            {'params': param, 'lr': 0.01} 
            for name, param in model.named_parameters() if 'scale_per_feat' in name
        ]
        other_params = [
            {'params': param} 
            for name, param in model.named_parameters() if 'scale_per_feat' not in name
        ]
        optimizer = optim.Adam([*scale_params, *other_params], lr=args.lr, weight_decay=args.weight_decay)
    elif args.model != 'svm' and args.model != 'ours':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = None
    
    return optimizer

def select_trainer(args, device, model, optimizer, data_loader_train, data_loader_test, adjacencies):
    MODEL = args.model
    
    if MODEL == 'svm':
        trainer = SVM_Trainer(args, device, model, data_loader_train, data_loader_test)
    elif MODEL == 'mlp':
        trainer = MLP_Trainer(args, device, model, optimizer, data_loader_train, data_loader_test, data_loader_test)
    elif MODEL == 'gcn' or MODEL == 'gat' or MODEL == 'gdc' or MODEL == 'adc' or MODEL == 'sgformer' or MODEL == 'nodeformer' or MODEL == 'difformer':
        trainer = GNN_Trainer(args, device, model, optimizer, data_loader_train, data_loader_test, data_loader_test)
    elif MODEL == 'graphheat':
        trainer = GraphHeat_Trainer(args, device, model, optimizer, data_loader_train, data_loader_test, data_loader_test, adjacencies[0].shape[0])
    elif MODEL == 'exact':
        trainer = Exact_Trainer(args, device, model, optimizer, data_loader_train, data_loader_test, data_loader_test, adjacencies[0].shape[0])
    elif MODEL == 'lsap':
        trainer = LSAP_Trainer(args, device, model, optimizer, data_loader_train, data_loader_test, data_loader_test, adjacencies[0].shape[0])
    elif MODEL == 'ours':
        trainer = OURS_Trainer(args, device, model, optimizer, data_loader_train, data_loader_test, data_loader_test)
    else:
        raise ValueError
    
    return trainer