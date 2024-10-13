import torch.nn.functional as F

from utils.utility import *
from utils.metric import *
from tqdm import tqdm

class GTAD_Trainer:
    def __init__(self, args, device, network, optimizer, train_loader, valid_loader, test_loader):
        self.args = args
        self.network = network
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = device
        
        self.checkpoint_path = "./checkpoints/" + str(args.model) + "/"
        
        if torch.cuda.is_available():
            self.network = self.network.to(device)

    def t_loss(self):
        alpha = 1
        n_feats = self.network.n_feats
        t_loss_final_ = []
        for i in range(n_feats):
            t_abs = torch.abs(self.network.scale_per_feat[i])
            t_zero = torch.zeros_like(self.network.scale_per_feat[i])

            t_loss_tmp = torch.where(self.network.scale_per_feat[i] < 1e-2, t_abs, t_zero)
            t_loss = torch.sum(t_loss_tmp) # λ|s|
            t_loss_final_.append(t_loss)
        
        t_loss_final = torch.mean(torch.stack(t_loss_final_))
        
        return alpha * t_loss_final
    
    def train(self, i):
        lt = [] # List of train loss
        at = [] # List of train accuracy
        
        acc_best = 0.0
        
        pbar = tqdm(range(1, self.args.epoch + 1))
        pbar.set_description(f'FOLD-{i}')
        for epoch in pbar:
            self.network.train()
            
            for _, X, Y, _, EIGVAL, EIGVEC in self.train_loader:
                self.optimizer.zero_grad() # Sets the gradients of all optimized tensors to zero
                
                Y_hat_train = self.network(X, EIGVAL, EIGVEC) # Shape: (sample, label)
                
                loss_train = F.nll_loss(Y_hat_train, Y) + self.t_loss()

                acc_train = compute_accuracy(Y_hat_train, Y)
                loss_train.backward()

                pbar.set_postfix({'loss': loss_train.item(), 'acc': acc_train.item()})
                
                self.optimizer.step() # Updates the parameters

            self.network.eval()
            with torch.no_grad():
                for _, X, Y, _, EIGVAL, EIGVEC in self.test_loader:
                    Y_hat_test = self.network(X, EIGVAL, EIGVEC) # Shape: (sample, label)
                    
                    acc_test = compute_accuracy(Y_hat_test, Y)
                    
                if acc_test > acc_best:
                    acc_best = acc_test
                    torch.save(self.network.state_dict(), 
                        self.checkpoint_path + 
                        str(self.args.feature) + '_' +
                        str(self.args.label) + '_' +
                        str(i) + '_' + 
                        'best_model.pth')

    def test(self, i):
        tl, ta, tac, tpr, tsp, tse, f1s = [[] for _ in range(7)]

        self.network.load_state_dict(torch.load(self.checkpoint_path + 
                                    str(self.args.feature) + '_' + 
                                    str(self.args.label) + '_' + 
                                    str(i) + '_' + 
                                    'best_model.pth'))
        # pdb.set_trace()
        self.network.eval()
        for _, X, Y, _, EIGVAL, EIGVEC in self.test_loader:
            Y_hat_test = self.network(X, EIGVAL, EIGVEC) # Shape: (sample, label)
            
            loss_test = F.nll_loss(Y_hat_test, Y)
            acc_test = compute_accuracy(Y_hat_test, Y)

            ac, pr, sp, se, f1 = confusion(Y_hat_test, Y)
            print(f"Acc: {ac:.3f} Pre: {pr:.3f} Spe: {sp:.3f} Rec: {se:.3f} F1s: {f1:.3f}")

            tl.append(loss_test.item())
            ta.append(acc_test.item())
            tac.append(ac.item())
            tpr.append(pr.item())
            tsp.append(sp.item())
            tse.append(se.item())
            f1s.append(f1.item())

        return np.mean(tl), np.mean(ta), np.mean(tac), np.mean(tpr), np.mean(tsp), np.mean(tse), np.mean(f1s), None