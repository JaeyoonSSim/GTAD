import torch
import pdb
from scipy.special import iv
from einops import rearrange

#######################################################################################
####################################### Exact #########################################
############################ Exact heat kernel computation ############################
#######################################################################################
def compute_heat_kernel(eigenvalue, eigenvector, t):
    hk_threshold = 1e-5
    n_samples = eigenvalue.shape[0]
    
    eigval = eigenvalue.type(torch.float)
    eigvec = eigenvector.type(torch.float)
    
    one = torch.ones_like(eigvec)
    ftr = torch.mul(one, torch.exp(-eigval).reshape(n_samples, 1, -1)) ** t.reshape(-1, 1) 
    hk = torch.matmul(torch.mul(eigvec, ftr), eigvec.transpose(-1,-2)) 
    hk[hk < hk_threshold] = 0
    
    hk_grad = torch.matmul(torch.matmul(torch.mul(eigvec, ftr), -torch.diag_embed(eigval)), eigvec.transpose(-1,-2))
    hk_sign = torch.where(hk >= hk_threshold, torch.ones_like(hk), torch.zeros_like(hk))  
    hk_grad = torch.mul(hk_grad, hk_sign)
    
    return hk, hk_grad



#######################################################################################
####################################### LSAP-C ########################################
########################## Chebyshev polynomial approximation #########################
#######################################################################################
def chebyshev_recurrence(Pf, Pf_old, L, b):
    out = (L @ Pf * 4) / b - (2 * Pf) - Pf_old
    out = 0.5 * (out.transpose(-1,-2) + out)
    
    return out

def compute_Tn(L, m, b_, device):
    num_samples = L.shape[0]
    
    fir = torch.zeros_like(L).to(device)
    tmp = torch.eye(L.shape[-1]).to(device)
    tmp = tmp.reshape((1, L.shape[-1], L.shape[-1]))
    sec = tmp.repeat(num_samples, 1, 1)
    
    T_n = [sec]
    tmp_ = torch.ones_like(L)
    b = b_.reshape(b_.shape[0],1,1) * tmp_
    T_0 = (L @ sec * 2) / b - sec - fir
    T_0 = 0.5 * (T_0.transpose(-1,-2) + T_0)
    fir, sec = sec, T_0
    
    for n in range(1, m + 1):
        T_n.append(sec)
        fir, sec = sec, chebyshev_recurrence(sec, fir, L, b)
    
    Tn = torch.stack(T_n)
    Tn = rearrange(Tn, 'd b n m -> b d n m')
    
    return Tn

def compute_heat_kernel_chebyshev(P_n, L, m, t, b_, device):
    hk_threshold = 1e-5
    
    t = t.reshape(-1, 1)
    b = b_.reshape(b_.shape[0],1,1)
    z = t * b / 2 
    ez = torch.exp(-z) 
    deg = torch.arange(0, m + 1).to(device) 
    ivd = iv(deg.cpu(), z.cpu()).to(device) 
    
    che = torch.zeros_like(L)
    coeff = 2 * ((-1) ** deg) * ez * ivd

    che_grad = torch.zeros_like(L)
    coeff_grad = b * ((-1) ** deg) * ez * ((iv(deg.cpu() - 1, z.cpu()).to(device) - (deg / z) * ivd) - ivd)
    
    for n in range(0, m + 1):
        che += coeff[:,:,n].unsqueeze(dim=-1) * P_n[:,n]
        che_grad += coeff_grad[:,:,n].unsqueeze(dim=-1) * P_n[:,n]
    
    che[che < hk_threshold] = 0
    che_sign = torch.where(che >= hk_threshold, torch.ones_like(che), torch.zeros_like(che))
    che_grad = torch.mul(che_grad, che_sign)
    
    return che, che_grad



#######################################################################################
####################################### LSAP-H ########################################
########################### Hermite polynomial approximation ##########################
#######################################################################################
def hermite_recurrence(n, Pf, Pf_old, L):
    out = (2 * L @ Pf) - (2 * n * Pf_old)
    out = 0.5 * (out.transpose(-1,-2) + out)
    return out

def compute_Hn(L, m, device):
    num_samples = L.shape[0]
    
    fir = torch.zeros_like(L).to(device)
    tmp = torch.eye(L.shape[-1]).to(device)
    tmp = tmp.reshape((1, L.shape[-1], L.shape[-1]))
    sec = tmp.repeat(num_samples, 1, 1)
    
    H_n = [sec]
    H_0 = 2 * L @ sec
    H_0 = 0.5 * (H_0.transpose(-1,-2) + H_0)
    fir, sec = sec, H_0
    
    for n in range(1, m + 1):
        H_n.append(sec)
        fir, sec = sec, hermite_recurrence(n, sec, fir, L)
    
    Hn = torch.stack(H_n)
    Hn = rearrange(Hn, 'd b n m -> b d n m')
    
    return Hn
    
def compute_heat_kernel_hermite(P_n, L, m, t, device):
    hk_threshold = 1e-5
    
    t = t.reshape(-1,1)
    deg = torch.arange(0, m + 1).to(device)
    z = (t ** 2) / 4
    ez = torch.exp(z)
    
    her = torch.zeros_like(L)
    coeff = ((-t / 2) ** deg) * ez
    
    her_grad = torch.zeros_like(L)
    coeff_grad = coeff * (t / 2) * ((2 * deg) / (t ** 2) + 1)
    
    her += coeff[:,0].unsqueeze(dim=-1) * P_n[:,0]
    her_grad += coeff_grad[:,0] * P_n[:,0]
    
    fac = 1
    for n in range(1, m + 1):
        fac *= n
        her += (1 / fac) * coeff[:,n].unsqueeze(dim=-1) * P_n[:,n]
        her_grad += (1 / fac) * coeff_grad[:,n] * P_n[:,n]
    
    her[her < hk_threshold] = 0
    
    her_sign = torch.where(her >= hk_threshold, torch.ones_like(her), torch.zeros_like(her))
    her_grad = torch.mul(her_grad, her_sign)
    
    return her, her_grad



#######################################################################################
####################################### LSAP-L ########################################
########################## Laguerre polynomial approximation ##########################
#######################################################################################
def laguerre_recurrence(n, Pf, Pf_old, L):
    out = (-L @ Pf + (2 * n + 1) * Pf - n * Pf_old) / (n + 1)
    out = 0.5 * (out.transpose(-1,-2) + out)
    
    return out

def compute_Ln(L, m, device):
    num_samples = L.shape[0]
    
    fir = torch.zeros_like(L).to(device)
    tmp = torch.eye(L.shape[-1]).to(device)
    tmp = tmp.reshape((1, L.shape[-1], L.shape[-1]))
    sec = tmp.repeat(num_samples, 1, 1)
    
    L_n = [sec]
    L_0 = sec - L @ sec
    L_0 = 0.5 * (L_0.transpose(-1,-2) + L_0)
    fir, sec = sec, L_0
    
    for n in range(1, m + 1):
        L_n.append(sec)
        fir, sec = sec, laguerre_recurrence(n, sec, fir, L)
    
    Ln = torch.stack(L_n)
    Ln = rearrange(Ln, 'd b n m -> b d n m')
    
    return Ln
    
def compute_heat_kernel_laguerre(P_n, L, m, t, device):
    hk_threshold = 1e-5

    t = t.reshape(-1,1)
    deg = torch.arange(0, m + 1).to(device)
    
    lag = torch.zeros_like(L)
    coeff = (t ** deg) / ((t + 1) ** (deg + 1))

    lag_grad = torch.zeros_like(L)
    coeff_grad = coeff * (deg - t) / (t * (t + 1))
        
    for n in range(0, m + 1):
        lag += coeff[:,n].unsqueeze(dim=-1) * P_n[:,n]
        lag_grad += coeff_grad[:,n] * P_n[:,n]
    
    lag[lag < hk_threshold] = 0
    
    lag_sign = torch.where(lag >= hk_threshold, torch.ones_like(lag), torch.zeros_like(lag))
    lag_grad = torch.mul(lag_grad, lag_sign)
    
    return lag, lag_grad