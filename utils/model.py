import torch
import torch.optim as optim

from utils.utility import *
from utils.loader import *
from utils.train import *
from models.gtad import GTAD

def select_model(args, n_roi, n_modal, Y):
    model = GTAD(adj_dim=n_roi,
                in_dim=n_modal,
                hid_dim=args.hid,
                out_dim=torch.max(Y).item() + 1,
                dropout=args.dr)
    
    return model
        
def select_optimizer(args, model):
    scale_params = [
        {'params': param, 'lr': 0.01} 
        for name, param in model.named_parameters() if 'scale_per_feat' in name
    ]
    other_params = [
        {'params': param} 
        for name, param in model.named_parameters() if 'scale_per_feat' not in name
    ]
    optimizer = optim.Adam([*scale_params, *other_params], lr=args.lr, weight_decay=args.wd)
    
    return optimizer

def select_trainer(args, device, model, optimizer, dl_train, dl_test):
    trainer = GTAD_Trainer(args, device, model, optimizer, dl_train, dl_test, dl_test)

    return trainer