#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This implementation is partially adapted from the work of Marios Mattheakis
on Hamiltonian Neural Networks for the Henon–Heiles system.

Original code reference:
https://github.com/mariosmat/hamiltonianNNetODEs
"""

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import grad
import matplotlib.pyplot as plt
import time
import copy
from os import path
import sys

from utils_HHsystem import symEuler, HH_exact, HHsolution, energy ,saveData
from sklearn.metrics import mean_squared_error
import os
from torch.nn.functional import softplus
from matplotlib.ticker import MaxNLocator  # 导入工具类
from scipy.io import savemat
os.makedirs("plots", exist_ok=True)
dtype=torch.float


# %matplotlib inline
plt. close('all')


# Define the sin() activation function
class mySin(torch.nn.Module):
    @staticmethod
    def forward(input):
        return torch.sin(input)
   
    
   #####################################
# Hamiltonian Neural Network (HNN) class
####################################


# Calculate the derivatice with auto-differention
def dfx(x,f):
    return grad([f], [x], grad_outputs=torch.ones(x.shape, dtype=dtype), create_graph=True)[0]

def perturbPoints(grid,t0,tf,sig=0.5):
#   stochastic perturbation of the evaluation points
#   force t[0]=t0  & force points to be in the t-interval
    delta_t = grid[1] - grid[0]  
    noise = delta_t * torch.randn_like(grid)*sig
    t = grid + noise
    t.data[2] = torch.ones(1,1)*(-1)
    t.data[t<t0]=t0 - t.data[t<t0]
    t.data[t>tf]=2*tf - t.data[t>tf]
    # t.data[0] = torch.ones(1,1)*t0
    t.requires_grad = False
    return t

    
def parametricSolutions(t, nn, X0, f_type='exp'):
    # parametric solutions
    t0, x0, y0, px0, py0, _ = X0[0],X0[1],X0[2],X0[3],X0[4],X0[5]
    N1, N2, N3, N4 = nn(t)
    dt =t-t0

    if f_type == 'exp':
        f = (1 - torch.exp(-dt))
    elif f_type == 'v2':
        a = softplus(nn.a_raw)
        b = softplus(nn.b_raw)
        # ensure shapes broadcast: a is (1,), dt is (N,1)
        f = (1 - torch.exp(-a * dt)) / (1 + b * torch.exp(-a * dt))
    elif f_type == 'tanh':
        m = softplus(nn.m_raw)
        f = torch.tanh(m * dt)
    else:
        raise ValueError(f"Unknown f_type: {f_type}")
    ###新增结束

    x_hat  = x0  + f*N1
    y_hat  = y0  + f*N2
    px_hat = px0 + f*N3
    py_hat = py0 + f*N4
    return x_hat, y_hat, px_hat, py_hat

def hamEqs_Loss(t,x,y,px,py,lam):
    # Define the loss function by Hamilton Eqs., write explicitely the Ham. Equations
    xd,yd,pxd,pyd= dfx(t,x),dfx(t,y),dfx(t,px),dfx(t,py)
    fx  = xd - px; 
    fy  = yd - py; 
    fpx = pxd + x + 2.*lam*x*y
    fpy = pyd + y + lam*(x.pow(2) - y.pow(2))
    Lx  = (fx.pow(2)).mean();  Ly  = (fy.pow(2)).mean(); 
    Lpx = (fpx.pow(2)).mean(); Lpy = (fpy.pow(2)).mean();
    L = Lx + Ly + Lpx + Lpy
    return L




def hamiltonian(x,y,px,py,lam):
    #returns the hamiltonian ham for Kinetic (K)  and Potential (V) Energies
    V = 0.5*(x**2 + y**2) + lam*(x**2*y - y**3/3)
    K = 0.5*(px**2+py**2)
    ham = K + V
    return ham


def hamiltonian_Loss(t,x,y,px,py,lam):
# Define the loss function as the time derivative of the hamiltonian
    xd,yd,pxd,pyd= dfx(t,x),dfx(t,y),dfx(t,px),dfx(t,py)
    ham = 0.5*(px.pow(2)+py.pow(2)+x.pow(2)+y.pow(2))+lam*(x.pow(2)*y-y.pow(3)/3)
    hx  = grad([ham], [x], grad_outputs=torch.ones(x.shape, dtype=dtype), create_graph=True)[0]
    hy  = grad([ham], [y], grad_outputs=torch.ones(y.shape, dtype=dtype), create_graph=True)[0]
    hpx = grad([ham], [px], grad_outputs=torch.ones(px.shape, dtype=dtype), create_graph=True)[0]
    hpy = grad([ham], [py], grad_outputs=torch.ones(py.shape, dtype=dtype), create_graph=True)[0]
    ht = hx*xd + hy*yd + hpx*pxd + hpy*pyd
    L = (ht.pow(2)).mean()
    return L


# NETWORK ARCHITECTURE

# A two hidden layer NN, 1 input & two output
class odeNet_HH_MM(torch.nn.Module):
    def __init__(self, D_hid=10, f_type='exp', a_init=2.0, b_init=0.5, m_init=3.0):##新增后3个参数
        super(odeNet_HH_MM,self).__init__()

        # Define the Activation
#         self.actF = torch.nn.Sigmoid()
        self.actF = mySin()

        # define layers
        self.Lin_1   = torch.nn.Linear(1, D_hid)
        self.Lin_2   = torch.nn.Linear(D_hid, D_hid)
        self.Lin_out = torch.nn.Linear(D_hid, 4)

        self.f_type = f_type
        if self.f_type == 'tanh':
            self.m_raw = torch.nn.Parameter(torch.tensor([np.log(np.exp(m_init)-1)], dtype=dtype))
        if self.f_type == 'v2':
            self.a_raw = torch.nn.Parameter(torch.tensor([np.log(np.exp(a_init) - 1)], dtype=dtype))
            self.b_raw= torch.nn.Parameter(torch.tensor([np.log(np.exp(b_init) - 1)], dtype=dtype))
    def forward(self,t):
        # layer 1
        l = self.Lin_1(t);    h = self.actF(l)
        # layer 2
        l = self.Lin_2(h);    h = self.actF(l)
        # output layer
        r = self.Lin_out(h)
        xN  = (r[:,0]).reshape(-1,1); yN  = (r[:,1]).reshape(-1,1)
        pxN = (r[:,2]).reshape(-1,1); pyN = (r[:,3]).reshape(-1,1)
        return xN, yN, pxN, pyN

# Train the NN
def run_odeNet_HH_MM(X0, tf, neurons, epochs, n_train,lr,
                     PATH= "models/model_HH", loadWeights=False,
                     minLoss=1e-5, f_type='exp', a_init=2.0, b_init=0.5, m_init=3.0):##新增后3
    fc0 = odeNet_HH_MM(neurons, f_type=f_type, a_init=a_init, b_init=b_init, m_init=m_init)

    fc1 =  copy.deepcopy(fc0) # fc1 is a deepcopy of the network with the lowest training loss
    # optimizer
    betas = [0.999, 0.9999]

    optimizer = optim.Adam(fc0.parameters(), lr=lr, betas=betas)
    if getattr(fc0, "f_type", None) == "tanh":
        fc0.m_raw.requires_grad = False
        print(">>> m is frozen during training.")
    if getattr(fc0, "f_type", None) == "v2":
        fc0.a_raw.requires_grad = False
        fc0.b_raw.requires_grad = False
        print(">>> a,b is frozen during training.")
    Loss_history = [];     Llim =  1
    Loss_erg_history= []
    # === record parameter trajectories ===新增
    alpha_list, beta_list, m_list = [], [], []

    t0=X0[0];
    x0, y0, px0, py0, lam = X0[1], X0[2], X0[3], X0[4], X0[5]
    # Initial Energy that should be convserved

    ham0 = hamiltonian(x0,y0,px0,py0,lam)
    grid = torch.linspace(t0, tf, n_train).reshape(-1,1)



## LOADING WEIGHTS PART if PATH file exists and loadWeights=True
    if path.exists(PATH) and loadWeights==True:
        checkpoint = torch.load(PATH)
        fc0.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        tt = checkpoint['epoch']
        Ltot = checkpoint['loss']
        fc0.train(); # or model.eval


## TRAINING ITERATION
    TeP0 = time.time()
    for tt in range(epochs):
# Perturbing the evaluation points & forcing t[0]=t0
        # t=perturbPoints(grid,t0,tf,sig=.03*tf)
        t=perturbPoints(grid,t0,tf,sig= 0.3*tf)
        t.requires_grad = True

#  Network solutions
        x,y,px,py =parametricSolutions(t,fc0,X0, f_type=f_type)###新增ftype

# LOSS FUNCTION
    #  Loss function defined by Hamilton Eqs.
        Ltot = hamEqs_Loss(t,x,y,px,py,lam)
        if tt % 1000 == 0:
            print(f"Epoch {tt}, Loss = {Ltot.item():.4e}")
    # ENERGY REGULARIZATION
        ham  = hamiltonian(x,y,px,py,lam)
        # L_erg =  ( ( ham - ham0).pow(2) ).mean()
        L_erg =  .5*( ( ham - ham0).pow(2) ).mean()
        Ltot=Ltot+ L_erg

# OPTIMIZER
        Ltot.backward(retain_graph=False)#True

# torch.nn.utils.clip_grad_norm_(fc0.parameters(), 1.0)
        optimizer.step()

# === record parameter evolution ===新增
        if f_type == 'v2':
            alpha_list.append(softplus(fc0.a_raw).item())
            beta_list.append(softplus(fc0.b_raw).item())

        elif f_type == 'tanh':
            m_list.append(softplus(fc0.m_raw).item())


        loss = Ltot.data.numpy()
        loss_erg=L_erg.data.numpy()
        optimizer.zero_grad()


# keep the loss function history
        Loss_history.append(loss)
        Loss_erg_history.append(loss_erg)

#Keep the best model (lowest loss) by using a deep copy
        if  tt > 0.8*epochs  and Ltot < Llim:
            fc1 =  copy.deepcopy(fc0)
            Llim=Ltot

# break the training after a thresold of accuracy
        if Ltot < minLoss :
            fc1 =  copy.deepcopy(fc0)
            print('Reach minimum requested loss')
            print(f"Reach minLoss={minLoss:.1e} at epoch {tt}")
            break



    TePf = time.time()
    runTime = TePf - TeP0


    if fc1.f_type == 'v2':
        print("a =", softplus(fc1.a_raw).item())
        print("b =", softplus(fc1.b_raw).item())

    elif fc1.f_type == 'tanh':
        print("m =", softplus(fc1.m_raw).item())

    # === Save parameter trajectories and final parameters ===
    save_dict = {
        'epoch': tt,
        'model_state_dict': fc1.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': Ltot,
    }

    if fc1.f_type == 'v2':
        save_dict['a_raw'] = fc1.a_raw.detach().cpu().numpy()
        save_dict['b_raw'] = fc1.b_raw.detach().cpu().numpy()
        save_dict['alpha_list'] = alpha_list
        save_dict['beta_list'] = beta_list

    elif fc1.f_type == 'tanh':
        save_dict['m_raw'] = fc1.m_raw.detach().cpu().numpy()
        save_dict['m_list'] = m_list

    torch.save(save_dict, PATH)###3
    return fc1, Loss_history, runTime, Loss_erg_history,alpha_list, beta_list, m_list

###

def trainModel(X0, t_max, neurons, epochs, n_train, lr,
               PATH="models/model_HH", loadWeights=False, minLoss=1e-6, showLoss=True, f_type='exp'):
    model,loss,runTime, loss_erg, alpha_list, beta_list, m_list = run_odeNet_HH_MM(X0, t_max, neurons, epochs, n_train,lr,PATH=PATH,  loadWeights=loadWeights, minLoss=minLoss, f_type=f_type)

    np.savetxt('data/loss.txt',loss)

    if showLoss==True :
        print('Training time (minutes):', runTime/60)
        print('Training Loss: ',  loss[-1] )
        plt.figure()
        plt.loglog(loss,'-b',alpha=0.975, label='Total loss');
        plt.loglog(loss_erg,'-r',alpha=0.75, label='Energy penalty');
        plt.legend()
        plt.tight_layout()
        plt.ylabel('Loss');plt.xlabel('t')

        plt.savefig(PATH.replace('models/', 'plots/').replace('.pt', '_loss.pdf'))

        loss_path = PATH.replace('.pt', '_loss.txt')
        np.savetxt(loss_path, loss)
    return model, loss, runTime, alpha_list, beta_list, m_list


def loadModel(PATH="models/model_HH", f_type='exp', neurons=None, a_init=2.0, b_init=0.5, m_init=3.0):
    if path.exists(PATH):
        fc0 = odeNet_HH_MM(neurons, f_type=f_type, a_init=a_init, b_init=b_init, m_init=m_init)#后俩新增
        checkpoint = torch.load(PATH)
        fc0.load_state_dict(checkpoint['model_state_dict'])
        fc0.train(); # or model.eval

        # === restore parameters ===
        if f_type == "v2":
            if "a_raw" in checkpoint:
                fc0.a_raw.data = torch.tensor(checkpoint["a_raw"], dtype=torch.float)
            if "b_raw" in checkpoint:
                fc0.b_raw.data = torch.tensor(checkpoint["b_raw"], dtype=torch.float)

        elif f_type == "tanh":
            if "m_raw" in checkpoint:
                fc0.m_raw.data = torch.tensor(checkpoint["m_raw"], dtype=torch.float)
        print(f"Model loaded from {PATH}. f_type={f_type}, neurons={neurons}")
    else:
        print('Warning: There is not any trained model. Terminate')
        sys.exit()

    return fc0    



############
# Set the initial state. lam controls the nonlinearity
t0, x0, y0, px0, py0, lam =  0, 0.3,-0.3, 0.3, 0.15, 1; 
X0 = [t0, x0, y0, px0, py0, lam]


# Run first a short time prediction. 
# Then load the model and train for longer time


t_max, N =  24*np.pi, 1000;
print(t_max * 0.069, ' Lyapunov times prediction'); dt = t_max/N;
n_train, neurons, epochs, lr = N, 80, int(5e4), 5e-3
# trainModel(X0, t_max, neurons, epochs, n_train, lr,  loadWeights=False, minLoss=1e-8, showLoss=True)



# # === Train exp model ===
# model_exp, loss_exp, time_exp , alpha_exp, beta_exp, m_exp= trainModel(
#     X0, t_max, neurons, epochs, n_train, lr,
#     loadWeights=False,
#     PATH="models/model_HH_exp.pt",
#     f_type='exp', minLoss=1e-8, showLoss=True
#
# )
# # #
# # # # === Train v2 model ===
# model_v2, loss_v2, time_v2, alpha_v2, beta_v2, m_v2 = trainModel(
#     X0, t_max, neurons, epochs, n_train, lr,
#     loadWeights=True,
#     PATH="models/model_HH_v2.pt",
#     f_type='v2', minLoss=1e-8, showLoss=True
# )
#
# # # # # === Train tanh model ===
# # model_tanh, loss_tanh, time_tanh, alpha_tanh, beta_tanh, m_tanh = trainModel(
# #     X0, t_max, neurons, epochs, n_train, lr,
# #     loadWeights=True,
# #     PATH="models/model_HH_tanh.pt",
# #     f_type='tanh', minLoss=1e-8, showLoss=True
# # )



#####################################
# TEST THE PREDICTED SOLUTIONS
#######################################3

nTest = N ; t_max_test = 1.0*t_max
tTest = torch.linspace(t0,t_max_test,nTest)

tTest = tTest.reshape(-1,1);
tTest.requires_grad=True
t_net = tTest.detach().numpy()

model_exp  = loadModel(PATH="models/model_HH_exp.pt", f_type='exp', neurons=80)
model_v2 = loadModel(PATH="models/model_HH_v2.pt", f_type='v2', neurons=80)
model_tanh = loadModel(PATH="models/model_HH_tanh.pt", f_type='tanh', neurons=80)

x_exp, y_exp, px_exp, py_exp = parametricSolutions(tTest, model_exp, X0, f_type='exp')
x_v2, y_v2, px_v2, py_v2 = parametricSolutions(tTest, model_v2, X0, f_type='v2')
x_tanh, y_tanh, px_tanh, py_tanh = parametricSolutions(tTest, model_tanh, X0, f_type='tanh')

x_exp = x_exp.detach().numpy(); y_exp = y_exp.detach().numpy()
px_exp = px_exp.detach().numpy(); py_exp = py_exp.detach().numpy()

x_v2 = x_v2.detach().numpy(); y_v2 = y_v2.detach().numpy()
px_v2 = px_v2.detach().numpy(); py_v2 = py_v2.detach().numpy()

x_tanh = x_tanh.detach().numpy(); y_tanh = y_tanh.detach().numpy()
px_tanh = px_tanh.detach().numpy(); py_tanh = py_tanh.detach().numpy()

E_exp = energy(x_exp, y_exp, px_exp, py_exp, lam)
E_v2 = energy(x_v2, y_v2, px_v2, py_v2, lam)
E_tanh = energy(x_tanh, y_tanh, px_tanh, py_tanh, lam)




# ####################
# Scipy solver
######################
t_num = np.linspace(t0, t_max_test, N)
E0, E_ex = HH_exact(N,x0, y0, px0, py0, lam)
x_num, y_num, px_num, py_num = HHsolution(N,t_num, x0, y0, px0, py0, lam)
E_num = energy(x_num, y_num, px_num, py_num, lam)



# ###################
# # Symplectic Euler
# # ####################
Ns = n_train -1; 
E_s, x_s, y_s, px_s, py_s, t_s = symEuler(Ns, x0,y0, px0,py0,t0,t_max_test,lam)
# # 10 times more time points

Ns10 = 10*n_train ; 

T0 = time.time()
E_s10, x_s10, y_s10, px_s10, py_s10, t_s10 = symEuler(Ns10, x0,y0, px0,py0,t0,t_max_test,lam)
runTimeEuler = time.time() - T0
print('Euler runtime is ', runTimeEuler/60)

################
# Make the plots
#################

# Figure for trajectories: x(t), p(t), energy in time E(t), 
#          and phase space trajectory p(x)

lineW = 2 # Line thicness

plt.figure(figsize=(16,10))
# 新增


########################
# plot1
plt.subplot(5, 1, 1)
plt.plot(t_num, E_ex, '-g', linewidth=lineW, label='Ground truth (SciPy)')
plt.plot(t_net, E_v2, '-.r', linewidth=lineW, label=r'$NN \  f(t)=\frac{1-e^{-\alpha t}}{1+\beta e^{-\alpha t}}$')
plt.ylabel('E')
plt.ylim([0.127, 0.1295])
plt.xlim([0, np.pi*24])
plt.legend(loc='lower right', ncol=2)


x_ticks = np.arange(0, 24*np.pi+0.001, np.pi)
plt.xticks(x_ticks)
plt.gca().set_xticklabels([])
y_ticks_1 = np.arange(0.127, 0.1295, 0.0005)
plt.yticks(y_ticks_1)
plt.gca().set_yticklabels([f'{y:.3f}' if y in [0.127, 0.128, 0.129] else '' for y in y_ticks_1])
plt.grid(True, linestyle='--', color='lightgray', alpha=0.6, linewidth=0.8)

# plot2
plt.subplot(5, 1, 2)
plt.plot(t_num, x_num, '-g', linewidth=lineW, label='Ground truth (SciPy)')
plt.plot(t_net, x_v2, '-.r', linewidth=lineW, label=r'$NN \  f(t)=\frac{1-e^{-\alpha t}}{1+\beta e^{-\alpha t}}$')
plt.ylim([-0.6, 0.6])
plt.xlim([0, np.pi*24])
plt.ylabel('x')
plt.xticks(x_ticks)
plt.gca().set_xticklabels([])
y_ticks_2 = np.arange(-0.6, 0.6 + 0.1, 0.1)
plt.yticks(y_ticks_2)
plt.gca().set_yticklabels(['0' if round(y, 1) == 0.0 else f'{y:.1f}' if round(y, 1) in [-0.5, 0.5] else ''
    for y in y_ticks_2])
plt.grid(True, linestyle='--', color='lightgray', alpha=0.6, linewidth=0.8)

# 3
plt.subplot(5, 1, 3)
plt.plot(t_num, y_num, '-g', linewidth=lineW, label='Ground truth (SciPy)')
plt.plot(t_net, y_v2, '-.r', linewidth=lineW, label=r'$NN \  f(t)=\frac{1-e^{-\alpha t}}{1+\beta e^{-\alpha t}}$')
plt.ylim([-0.6, 0.6])
plt.xlim([0, np.pi*24])
plt.ylabel('y')
plt.xticks(x_ticks)
plt.gca().set_xticklabels([])
plt.yticks(y_ticks_2)
plt.gca().set_yticklabels(['0' if round(y, 1) == 0.0 else f'{y:.1f}' if round(y, 1) in [-0.5, 0.5] else ''
    for y in y_ticks_2])
plt.grid(True, linestyle='--', color='lightgray', alpha=0.6, linewidth=0.8)

# 4
plt.subplot(5, 1, 4)
plt.plot(t_num, px_num, '-g', linewidth=lineW, label='Ground truth (SciPy)')
plt.plot(t_net, px_v2, '-.r', linewidth=lineW, label=r'$NN \  f(t)=\frac{1-e^{-\alpha t}}{1+\beta e^{-\alpha t}}$')
plt.ylim([-0.6, 0.6])
plt.xlim([0, np.pi*24])
plt.ylabel('$p_x$')
plt.xticks(x_ticks)
plt.gca().set_xticklabels([])
plt.yticks(y_ticks_2)
plt.gca().set_yticklabels(['0' if round(y, 1) == 0.0 else f'{y:.1f}' if round(y, 1) in [-0.5, 0.5] else ''
    for y in y_ticks_2])
plt.grid(True, linestyle='--', color='lightgray', alpha=0.6, linewidth=0.8)

plt.subplot(5, 1, 5)
plt.plot(t_num, py_num, '-g', linewidth=lineW, label='Ground truth (SciPy)')
plt.plot(t_net, py_v2, '-.r', linewidth=lineW, label=r'$NN \  f(t)=\frac{1-e^{-\alpha t}}{1+\beta e^{-\alpha t}}$')
plt.ylabel('$p_y$')
plt.xlabel('Time')
plt.ylim([-0.6, 0.6])
plt.xlim([0, np.pi*24])
x_ticks_5 = np.linspace(0, 24 * np.pi, 13)
plt.xticks(x_ticks)
plt.gca().set_xticklabels([f'{x/np.pi:.0f}π' if x in x_ticks_5 else '' for x in x_ticks])
plt.yticks(y_ticks_2)
plt.gca().set_yticklabels(['0' if round(y, 1) == 0.0 else f'{y:.1f}' if round(y, 1) in [-0.5, 0.5] else ''
    for y in y_ticks_2])
plt.grid(True, linestyle='--', color='lightgray', alpha=0.6, linewidth=0.8)

# plt.subplots_adjust(hspace=0.15)
plt.savefig('plots/HenonHeiles_trajectories.pdf', bbox_inches='tight')  # 添加bbox_inches防止标签被裁剪


##
dx_exp = x_num - x_exp[:,0]
dy_exp = y_num - y_exp[:,0]
dpx_exp = px_num - px_exp[:,0]
dpy_exp = py_num - py_exp[:,0]

dx_v2 = x_num - x_v2[:,0]
dy_v2 = y_num - y_v2[:,0]
dpx_v2 = px_num - px_v2[:,0]
dpy_v2 = py_num - py_v2[:,0]

dx_tanh = x_num - x_tanh[:,0]
dy_tanh = y_num - y_tanh[:,0]
dpx_tanh = px_num - px_tanh[:,0]
dpy_tanh = py_num - py_tanh[:,0]
# # calculate the errors for the solutions obtained by Euler
x_numN, y_numN, px_numN, py_numN   = HHsolution(Ns,t_s, x0, y0, px0, py0, lam)
dx_s = x_numN - x_s;        dpx_s = px_numN - px_s
dy_s = y_numN - y_s;        dpy_s = py_numN - py_s

x_numN, y_numN, px_numN, py_numN   = HHsolution(Ns10,t_s10, x0, y0, px0, py0, lam)
dx_s10 =  x_numN - x_s10;      dpx_s10 = px_numN - px_s10
dy_s10 = y_numN -  y_s10;       dpy_s10 = py_numN - py_s10


plt.figure(figsize=(10,8))

# (1) Δpx vs Δx (phase error scatter)
plt.subplot(2,2,1)
# plt.plot(dx_exp, dpx_exp, '--b', label=r'$NN \  f(t)=1-e^{-t}$')
# plt.plot(dx_tanh, dpx_tanh, '--m', label=r'$NN \ f(t)=\tanh(mt)$')
plt.plot(dx_v2, dpx_v2, '-.r', label=r'$NN \  f(t)=\frac{1-e^{-\alpha t}}{1+\beta e^{-\alpha t}}$')
plt.ylabel(r'$\delta_{p_x}$'); plt.xlabel(r'$\delta_x$')
plt.title('Phase error in x-dimension')
# plt.legend()
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))  # nbins=5 表示最多 5 个刻度
# plt.xlim(-0.002, 0.002)
# plt.ylim(-0.002, 0.002)
# (2) Δpy vs Δy (phase error scatter)
plt.subplot(2,2,2)
# plt.plot(dy_exp, dpy_exp, '--b', label=r'$NN \  f(t)=1-e^{-t}$')
# plt.plot(dy_v2, dpy_v2, '-.r', label='NN new f(t)')
# plt.plot(dy_tanh, dpy_tanh, '--m', label=r'$NN \ f(t)=\tanh(mt)$')
plt.plot(dy_v2, dpy_v2, '-.r', label=r'$NN \  f(t)=\frac{1-e^{-\alpha t}}{1+\beta e^{-\alpha t}}$')
plt.ylabel(r'$\delta_{p_y}$'); plt.xlabel(r'$\delta_y$')
plt.title('Phase error in y-dimension')
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))  # nbins=5 表示最多 5 个刻度
plt.legend()
# plt.xlim(-0.002, 0.002)
# plt.ylim(-0.002, 0.002)
# (3) Δx(t)
plt.subplot(2,2,3)
# plt.plot(t_num, dx_exp, '--b', label=r'$NN \  f(t)=1-e^{-t}$')
# plt.plot(t_num, dx_v2, '-.r', label='NN new f(t)')
# plt.plot(t_num, dx_tanh, '--m', label=r'$NN \ f(t)=\tanh(mt)$')
plt.plot(t_num, dx_v2, '-.r', label=r'$NN \  f(t)=\frac{1-e^{-\alpha t}}{1+\beta e^{-\alpha t}}$')
plt.ylabel(r'$\delta_x$'); plt.xlabel('t')
# plt.legend()
# plt.ylim(-0.002, 0.002)
# (4) Δy(t)
plt.subplot(2,2,4)
# plt.plot(t_num, dy_exp, '--b', label=r'$NN \  f(t)=1-e^{-t}$')
# plt.plot(t_num, dy_v2, '-.r', label='NN new f(t)')
# plt.plot(t_num, dy_tanh, '--m', label=r'$NN \ f(t)=\tanh(mt)$')
plt.plot(t_num, dy_v2, '-.r', label=r'$NN \  f(t)=\frac{1-e^{-\alpha t}}{1+\beta e^{-\alpha t}}$')
plt.ylabel(r'$\delta_y$'); plt.xlabel('t')
# plt.legend()
# plt.ylim(-0.002, 0.002)
# plt.tight_layout()

plt.savefig('plots/HenonHeiles_error_compare.pdf')
# plt.show()


# === Ground Truth（SciPy） ===
x_gt  = x_num.reshape(-1,1)
y_gt  = y_num.reshape(-1,1)
px_gt = px_num.reshape(-1,1)
py_gt = py_num.reshape(-1,1)

# === 误差函数 ===
def compute_errors(pred, gt):
    L2  = np.linalg.norm(pred - gt)
    MSE = mean_squared_error(gt, pred)
    return L2, MSE

# ----------------------------------------------------------
# exp
L2_x_exp,  MSE_x_exp  = compute_errors(x_exp,  x_gt)
L2_y_exp,  MSE_y_exp  = compute_errors(y_exp,  y_gt)
L2_px_exp, MSE_px_exp = compute_errors(px_exp, px_gt)
L2_py_exp, MSE_py_exp = compute_errors(py_exp, py_gt)

# ----------------------------------------------------------
# tanh
L2_x_tanh,  MSE_x_tanh  = compute_errors(x_tanh,  x_gt)
L2_y_tanh,  MSE_y_tanh  = compute_errors(y_tanh,  y_gt)
L2_px_tanh, MSE_px_tanh = compute_errors(px_tanh, px_gt)
L2_py_tanh, MSE_py_tanh = compute_errors(py_tanh, py_gt)

# ----------------------------------------------------------
# v2
L2_x_v2,  MSE_x_v2  = compute_errors(x_v2,  x_gt)
L2_y_v2,  MSE_y_v2  = compute_errors(y_v2,  y_gt)
L2_px_v2, MSE_px_v2 = compute_errors(px_v2, px_gt)
L2_py_v2, MSE_py_v2 = compute_errors(py_v2, py_gt)

# ----------------------------------------------------------
# Symplectic Euler

from scipy.interpolate import interp1d

interp_x_s  = interp1d(t_s,  x_s,  kind='linear', fill_value="extrapolate")
interp_y_s  = interp1d(t_s,  y_s,  kind='linear', fill_value="extrapolate")
interp_px_s = interp1d(t_s, px_s,  kind='linear', fill_value="extrapolate")
interp_py_s = interp1d(t_s, py_s,  kind='linear', fill_value="extrapolate")

x_euler  = interp_x_s(t_num).reshape(-1,1)
y_euler  = interp_y_s(t_num).reshape(-1,1)
px_euler = interp_px_s(t_num).reshape(-1,1)
py_euler = interp_py_s(t_num).reshape(-1,1)

L2_x_euler,  MSE_x_euler  = compute_errors(x_euler,  x_gt)
L2_y_euler,  MSE_y_euler  = compute_errors(y_euler,  y_gt)
L2_px_euler, MSE_px_euler = compute_errors(px_euler, px_gt)
L2_py_euler, MSE_py_euler = compute_errors(py_euler, py_gt)

# ----------------------------------------------------------

print("\n===== L2 Errors =====")
print(f"x   : exp={L2_x_exp:.3e}, tanh={L2_x_tanh:.3e}, v2={L2_x_v2:.3e}, Euler={L2_x_euler:.3e}")
print(f"y   : exp={L2_y_exp:.3e}, tanh={L2_y_tanh:.3e}, v2={L2_y_v2:.3e}, Euler={L2_y_euler:.3e}")
print(f"px  : exp={L2_px_exp:.3e}, tanh={L2_px_tanh:.3e}, v2={L2_px_v2:.3e}, Euler={L2_px_euler:.3e}")
print(f"py  : exp={L2_py_exp:.3e}, tanh={L2_py_tanh:.3e}, v2={L2_py_v2:.3e}, Euler={L2_py_euler:.3e}")

print("\n===== MSE Errors =====")
print(f"x   : exp={MSE_x_exp:.3e}, tanh={MSE_x_tanh:.3e}, v2={MSE_x_v2:.3e}, Euler={MSE_x_euler:.3e}")
print(f"y   : exp={MSE_y_exp:.3e}, tanh={MSE_y_tanh:.3e}, v2={MSE_y_v2:.3e}, Euler={MSE_y_euler:.3e}")
print(f"px  : exp={MSE_px_exp:.3e}, tanh={MSE_px_tanh:.3e}, v2={MSE_px_v2:.3e}, Euler={MSE_px_euler:.3e}")
print(f"py  : exp={MSE_py_exp:.3e}, tanh={MSE_py_tanh:.3e}, v2={MSE_py_v2:.3e}, Euler={MSE_py_euler:.3e}")
with open("results_all_models.txt", "w") as f:
    f.write("=== Error comparison for all loaded models ===\n\n")

    # ---------------- HNN exp ----------------
    f.write("Model: HNN  (f(t) = 1 - exp(-t))\n")
    f.write(f"L2_x   = {L2_x_exp:.6e}\n")
    f.write(f"L2_y   = {L2_y_exp:.6e}\n")
    f.write(f"L2_px  = {L2_px_exp:.6e}\n")
    f.write(f"L2_py  = {L2_py_exp:.6e}\n")
    f.write(f"MSE_x  = {MSE_x_exp:.6e}\n")
    f.write(f"MSE_y  = {MSE_y_exp:.6e}\n")
    f.write(f"MSE_px = {MSE_px_exp:.6e}\n")
    f.write(f"MSE_py = {MSE_py_exp:.6e}\n\n")

    # ---------------- TAHNN tanh ----------------
    f.write("Model: TAHNN  (f(t) = tanh(m t))\n")
    f.write(f"L2_x   = {L2_x_tanh:.6e}\n")
    f.write(f"L2_y   = {L2_y_tanh:.6e}\n")
    f.write(f"L2_px  = {L2_px_tanh:.6e}\n")
    f.write(f"L2_py  = {L2_py_tanh:.6e}\n")
    f.write(f"MSE_x  = {MSE_x_tanh:.6e}\n")
    f.write(f"MSE_y  = {MSE_y_tanh:.6e}\n")
    f.write(f"MSE_px = {MSE_px_tanh:.6e}\n")
    f.write(f"MSE_py = {MSE_py_tanh:.6e}\n\n")

    # ---------------- TAHNN rational ----------------
    f.write("Model: TAHNN  (f(t) = (1-exp(-αt))/(1+β exp(-αt)))\n")
    f.write(f"L2_x   = {L2_x_v2:.6e}\n")
    f.write(f"L2_y   = {L2_y_v2:.6e}\n")
    f.write(f"L2_px  = {L2_px_v2:.6e}\n")
    f.write(f"L2_py  = {L2_py_v2:.6e}\n")
    f.write(f"MSE_x  = {MSE_x_v2:.6e}\n")
    f.write(f"MSE_y  = {MSE_y_v2:.6e}\n")
    f.write(f"MSE_px = {MSE_px_v2:.6e}\n")
    f.write(f"MSE_py = {MSE_py_v2:.6e}\n\n")

    # ---------------- Symplectic Euler ----------------
    f.write("Model: Symplectic Euler\n")
    f.write(f"L2_x   = {L2_x_euler:.6e}\n")
    f.write(f"L2_y   = {L2_y_euler:.6e}\n")
    f.write(f"L2_px  = {L2_px_euler:.6e}\n")
    f.write(f"L2_py  = {L2_py_euler:.6e}\n")
    f.write(f"MSE_x  = {MSE_x_euler:.6e}\n")
    f.write(f"MSE_y  = {MSE_y_euler:.6e}\n")
    f.write(f"MSE_px = {MSE_px_euler:.6e}\n")
    f.write(f"MSE_py = {MSE_py_euler:.6e}\n\n")

print(">>> All model errors saved to results_all_models.txt")

# np.savetxt("results_exp.txt", [
#     ["L2_x", L2_x_exp],
#     ["L2_y", L2_y_exp],
#     ["L2_px", L2_px_exp],
#     ["L2_py", L2_py_exp],
#     ["MSE_x", MSE_x_exp],
#     ["MSE_y", MSE_y_exp],
#     ["MSE_px", MSE_px_exp],
#     ["MSE_py", MSE_py_exp],
# ], fmt="%s")
# np.savetxt("results_tanh.txt", [
#     ["L2_x", L2_x_tanh],
#     ["L2_y", L2_y_tanh],
#     ["L2_px", L2_px_tanh],
#     ["L2_py", L2_py_tanh],
#     ["MSE_x", MSE_x_tanh],
#     ["MSE_y", MSE_y_tanh],
#     ["MSE_px", MSE_px_tanh],
#     ["MSE_py", MSE_py_tanh],
#     ["m_final", m_tanh[-1]],
# ], fmt="%s")
# np.savetxt("results_v2.txt", [
#     ["L2_x", L2_x_v2],
#     ["L2_y", L2_y_v2],
#     ["L2_px", L2_px_v2],
#     ["L2_py", L2_py_v2],
#     ["MSE_x", MSE_x_v2],
#     ["MSE_y", MSE_y_v2],
#     ["MSE_px", MSE_px_v2],
#     ["MSE_py", MSE_py_v2],
#     ["alpha_final", alpha_v2[-1]],
#     ["beta_final", beta_v2[-1]],
# ], fmt="%s")


#loss
plt.figure(figsize=(6,5))

paths = [
    "models/model_HH_exp_loss.txt",
    "models/model_HH_v2_loss.txt",
    "models/model_HH_tanh_loss.txt"
]
labels = [
    r'$f(t)=1-e^{-t}$',
    r'$f(t)=\frac{1-e^{-\alpha t}}{1+\beta e^{-\alpha t}}$',
    r'$f(t)=\tanh(mt)$'
]
styles = ["-b"
    , "-r", "-m"
          ]

for p, label, style in zip(paths, labels, styles):
    if os.path.exists(p):
        loss = np.loadtxt(p)
        plt.semilogy(loss, style, linewidth=2, label=label)
    else:
        print(f"Warning: {p} not found, skip.")

plt.xlabel("Epoch")
plt.ylabel("Loss")
# plt.title("Loss comparison between f(t) forms")
plt.legend()
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.savefig("plots/loss.pdf")

# === plot parameter evolution for v2
# plt.plot(alpha_v2, label='alpha')
# plt.plot(beta_v2, label='beta')
# plt.xlabel("Epoch")
# plt.ylabel("Value")
# plt.title("Evolution of alpha & beta")
# plt.legend()
# plt.savefig("plots/param_evolution_a_b.pdf")
# plt.figure()
# plt.plot(m_tanh, label='m')
# plt.xlabel("Epoch")
# plt.ylabel("Value")
# plt.title("Evolution of m")
# plt.legend()
# plt.savefig("plots/param_evolution_m.pdf")

mat_dir = "mat_data"
# savemat(os.path.join(mat_dir, "HH_transfer.mat"), {
#
#     # -------- time --------
#     "t_num": t_num,
#     "t_net": t_net,
#     "t_s": t_s,
#     "t_s10": t_s10,
#
#     # -------- ground truth (SciPy) --------
#     "x_num": x_num,
#     "y_num": y_num,
#     "px_num": px_num,
#     "py_num": py_num,
#     "E_ex": E_ex,
#     "E_num": E_num,
#
#     # -------- NN: exp --------
#     "x_exp": x_exp,
#     "y_exp": y_exp,
#     "px_exp": px_exp,
#     "py_exp": py_exp,
#     "E_exp": E_exp,
#
#     # -------- NN: tanh --------
#     "x_tanh": x_tanh,
#     "y_tanh": y_tanh,
#     "px_tanh": px_tanh,
#     "py_tanh": py_tanh,
#     "E_tanh": E_tanh,
#
#     # -------- NN: v2 --------
#     "x_v2": x_v2,
#     "y_v2": y_v2,
#     "px_v2": px_v2,
#     "py_v2": py_v2,
#     "E_v2": E_v2,
#
#     # -------- Symplectic Euler --------
#     "x_s": x_s,
#     "y_s": y_s,
#     "px_s": px_s,
#     "py_s": py_s,
#     "E_s": E_s,
#
#     # -------- Symplectic Euler x10 --------
#     "x_s10": x_s10,
#     "y_s10": y_s10,
#     "px_s10": px_s10,
#     "py_s10": py_s10,
#     "E_s10": E_s10,
#
#     # -------- errors (NN vs GT) --------
#     "dx_exp": dx_exp,
#     "dy_exp": dy_exp,
#     "dpx_exp": dpx_exp,
#     "dpy_exp": dpy_exp,
#
#     "dx_tanh": dx_tanh,
#     "dy_tanh": dy_tanh,
#     "dpx_tanh": dpx_tanh,
#     "dpy_tanh": dpy_tanh,
#
#     "dx_v2": dx_v2,
#     "dy_v2": dy_v2,
#     "dpx_v2": dpx_v2,
#     "dpy_v2": dpy_v2,
# "dx_s": dx_s,   "dy_s": dy_s,
#     "dpx_s": dpx_s, "dpy_s": dpy_s,
#
#     "dx_s10": dx_s10,   "dy_s10": dy_s10,
#     "dpx_s10": dpx_s10, "dpy_s10": dpy_s10
# })
# savemat(os.path.join(mat_dir, "HH_transfer_base.mat"), {
#
#     # -------- NN: exp --------
#     "x_base": x_exp,
#     "y_base": y_exp,
#     "px_base": px_exp,
#     "py_base": py_exp,
#     "E_base": E_exp,
#
#
#     # -------- errors (NN vs GT) --------
#     "dx_base": dx_exp,
#     "dy_base": dy_exp,
#     "dpx_base": dpx_exp,
#     "dpy_base": dpy_exp,
# })
# print("✔ Data saved")
plt.show()