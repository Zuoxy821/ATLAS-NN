#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This implementation is partially adapted from the work of Marios Mattheakis
on Hamiltonian Neural Networks for the nonlinear oscillator system.

Original code reference:
https://github.com/mariosmat/hamiltonianNNetODEs
"""

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import grad
# import matplotlib
import matplotlib.pyplot as plt
import time
import copy
# from scipy.integrate import odeint
from os import path
from sklearn.metrics import mean_squared_error
from utils_NLoscillator import symEuler, NLosc_exact, NLosc_solution, energy ,saveData
import os##
from torch.nn.functional import softplus
from scipy.io import savemat
os.makedirs("plots", exist_ok=True)##

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

# Define some functions used by the Hamiltonian network
def parametricSolutions(t, nn, X0, f_type='exp'):
    # parametric solutions
    t0, x0, px0, _ = X0[0],X0[1],X0[2],X0[3]
    N1, N2  = nn(t)
    dt =t-t0

    # f = (1-torch.exp(-dt))
    if f_type == 'exp':
        f = (1 - torch.exp(-dt))
    elif f_type == 'v2':
        a = softplus(nn.a_raw)
        b = softplus(nn.b_raw)
        f = (1 - torch.exp(-a * dt)) / (1 + b * torch.exp(-a * dt))
    elif f_type == 'tanh':
        m = softplus(nn.m_raw)
        f = torch.tanh(m * dt)
    else:
        raise ValueError(f"Unknown f_type: {f_type}")
    ###新增结束
    x_hat  = x0  + f*N1 
    px_hat = px0 + f*N2 
    return x_hat, px_hat

def hamEqs_Loss(t,x,px,lam):
    # Define the loss function by Hamilton Eqs., write explicitely the Ham. Equations
    xd,pxd= dfx(t,x),dfx(t,px)
    fx  = xd - px; 
    fpx = pxd + x + lam*x.pow(3)
    Lx  = (fx.pow(2)).mean();     Lpx = (fpx.pow(2)).mean();
    L = Lx  + Lpx
    return L


def hamiltonian(x,px,lam):
    #returns the hamiltonian ham for Kinetic (K)  and Potential (V) Energies
    K = 0.5*px**2
    V = 0.5*x**2  + lam*x**4/4           
    ham = K + V
    return ham
    


# NETWORK ARCHITECTURE
    
# A two hidden layer NN, 1 input & two output
class odeNet_NLosc_MM(torch.nn.Module):
    def __init__(self, D_hid=10, f_type='exp', a_init=1.0, b_init=1.0, m_init=1.0): ### 加上ftpe参数
        super(odeNet_NLosc_MM,self).__init__()

#####    CHOOCE THE ACTIVATION FUNCTION
        self.actF = mySin()
        # self.actF = torch.nn.Sigmoid()

# define layers
        self.Lin_1   = torch.nn.Linear(1, D_hid)
        self.Lin_2   = torch.nn.Linear(D_hid, D_hid)
        self.Lin_out = torch.nn.Linear(D_hid, 2)

        self.f_type = f_type
        if self.f_type == 'tanh':
            self.m_raw = torch.nn.Parameter(torch.tensor([np.log(np.exp(m_init)-1)], dtype=dtype))
        if self.f_type == 'v2':
            self.a_raw = torch.nn.Parameter(torch.tensor([np.log(np.exp(a_init) - 1)], dtype=dtype))
            self.b_raw = torch.nn.Parameter(torch.tensor([np.log(np.exp(b_init) - 1)], dtype=dtype))
    def forward(self,t):
        # layer 1
        l = self.Lin_1(t);    h = self.actF(l)
        # layer 2
        l = self.Lin_2(h);    h = self.actF(l)

        # output layer
        r = self.Lin_out(h)
        xN  = (r[:,0]).reshape(-1,1); pxN = (r[:,1]).reshape(-1,1);
        return xN, pxN



# FUNCTION NETWORK TRAINING
def run_odeNet_NLosc_MM(X0, tf, neurons, epochs, n_train,lr,
                        PATH= "models/model_NL", loadWeights=False,
                     minLoss=1e-5, f_type='exp', a_init=1.0, b_init=1.0, m_init=1.0):

    fc0 = odeNet_NLosc_MM(neurons, f_type=f_type, a_init=a_init, b_init=b_init, m_init=m_init)## 修改后fc0
    fc1 =  copy.deepcopy(fc0) #fc1 is a deepcopy of the network with the lowest training loss
    # optimizer
    betas = [0.999, 0.9999]
    optimizer = optim.Adam(fc0.parameters(), lr=lr, betas=betas)
    if f_type == 'tanh':
        fc0.m_raw.requires_grad = False
        print(">>> m is frozen during training.")
    if f_type == 'v2':
        fc0.a_raw.requires_grad = False
        fc0.b_raw.requires_grad = False
        print(">>> a,b is frozen during training.")
    Loss_history = [];
    Loss_ham_history = []
    Loss_energy_history = []
    Llim =  1
    # === record parameter trajectories ===新增
    alpha_list, beta_list, m_list = [], [], []

    t0=X0[0];
    x0,px0,lam=X0[1],X0[2],X0[3]

    # Compute the initial energy ham0 that should remain constant
    ham0 = hamiltonian(x0, px0, lam)

    grid=torch.linspace(t0, tf,  n_train).reshape(-1,1)

## LOADING WEIGHTS PART if PATH file exists and loadWeights=True
    if path.exists(PATH) and loadWeights==True:
        checkpoint = torch.load(PATH)
        fc0.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        tt = checkpoint['epoch']
        Ltot = checkpoint['loss']
        fc0.train(); # or model.eval

##  TRAINING ITERATION
    TeP0 = time.time()

    for tt in range(epochs):
# Perturbing the evaluation points & forcing t[0]=t0
        if tt%2000==0:
            print(tt)
        t= perturbPoints(grid,t0,tf,sig=0.03*tf)
        t.requires_grad = True
#  Network solutions
        x,px =parametricSolutions(t,fc0,X0, f_type=f_type)

# LOSS FUNCTION
    #  Loss function defined by Hamilton Eqs.
        L_ham = hamEqs_Loss(t, x, px, lam)

        # Energy regularization
        ham = hamiltonian(x, px, lam)
        L_energy = ((ham - ham0).pow(2)).mean()

        # Total loss
        Ltot = L_ham +L_energy

# OPTIMIZER
        Ltot.backward(retain_graph=False)#True
        optimizer.step()
    # === record parameter evolution
        if f_type == 'v2':
            alpha_list.append(softplus(fc0.a_raw).item())
            beta_list.append(softplus(fc0.b_raw).item())

        elif f_type == 'tanh':
            m_list.append(softplus(fc0.m_raw).item())

        loss = Ltot.data.numpy()
        optimizer.zero_grad()


# keep the loss function history
        Loss_history.append(loss)
        Loss_ham_history.append(L_ham.detach().cpu().item())
        Loss_energy_history.append(L_energy.detach().cpu().item())

#Keep the best model (lowest loss) by using a deep copy
        if  tt > 0.8*epochs  and Ltot < Llim:
            fc1 =  copy.deepcopy(fc0)
            Llim=Ltot


# break the training after a thresold of accuracy
        if Ltot < minLoss :
            fc1 =  copy.deepcopy(fc0)
            print(f"Reach minLoss={minLoss:.1e} at epoch {tt}")
            break




    TePf = time.time()
    runTime = TePf - TeP0


    if fc1.f_type == 'v2':
        print("a =", softplus(fc1.a_raw).item())
        print("b =", softplus(fc1.b_raw).item())

    elif fc1.f_type == 'tanh':
        print("m =", softplus(fc1.m_raw).item())


    save_dict = {
        'epoch': tt,
        'model_state_dict': fc1.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # === loss histories ===
        'Loss_total_history': Loss_history,
        'Loss_ham_history': Loss_ham_history,
        'Loss_energy_history': Loss_energy_history,
    }
    # === save f(t) parameters ===
    if fc1.f_type == 'v2':
        save_dict['a_raw'] = fc1.a_raw.detach().cpu().numpy()
        save_dict['b_raw'] = fc1.b_raw.detach().cpu().numpy()
        save_dict['alpha_list'] = alpha_list
        save_dict['beta_list'] = beta_list

    elif fc1.f_type == 'tanh':
        save_dict['m_raw'] = fc1.m_raw.detach().cpu().numpy()
        save_dict['m_list'] = m_list

    torch.save(save_dict, PATH)
    return fc1, Loss_history, Loss_ham_history, Loss_energy_history,runTime,alpha_list, beta_list, m_list


# TRAINING FUNCTION
def trainModel(X0, t_max, neurons, epochs, n_train, lr, PATH="models/model_NL",  loadWeights=False, minLoss=1e-5, showLoss=True, f_type='exp'):
    model,loss, loss_ham,loss_energy,runTime, alpha_list, beta_list, m_list = run_odeNet_NLosc_MM(X0, t_max, neurons, epochs, n_train, lr,  PATH=PATH,   loadWeights=loadWeights, minLoss=minLoss, f_type=f_type)
###新增PATH
    # ===== save losses (one-to-one with model file) =====
    base = PATH.replace('.pt', '')

    np.savetxt(base + '_loss_total.txt', loss)
    np.savetxt(base + '_loss_ham.txt', loss_ham)
    np.savetxt(base + '_loss_energy.txt', loss_energy)


    if showLoss:
            print('Training time (minutes):', runTime / 60)

            plt.figure()
            plt.loglog(loss, label='Total loss')
            plt.loglog(loss_ham, label='Hamilton residual')
            plt.loglog(loss_energy, label='Energy error')
            plt.legend()
            plt.tight_layout()

            plt.savefig(base + '_loss.pdf')
            plt.close()
    return model, loss,  loss_ham,loss_energy, runTime, alpha_list, beta_list, m_list

def loadModel(PATH="models/model_NL", f_type='exp', neurons=None, a_init=1.0, b_init=1.0, m_init=1.0):#后面三个都是加的
    fc0 = odeNet_NLosc_MM(neurons, f_type=f_type, a_init=a_init,b_init=b_init, m_init=m_init)#
    checkpoint = torch.load(PATH)
    fc0.load_state_dict(checkpoint['model_state_dict'])
    fc0.train(); # or model.eval

    ##新增
    # === load f(t) parameters ===
    if f_type == "v2":
        if "a_raw" in checkpoint:
            fc0.a_raw.data = torch.tensor(checkpoint["a_raw"], dtype=torch.float)
        if "b_raw" in checkpoint:
            fc0.b_raw.data = torch.tensor(checkpoint["b_raw"], dtype=torch.float)

        alpha_list = checkpoint.get("alpha_list", [])
        beta_list = checkpoint.get("beta_list", [])

        print(f"Loaded a={softplus(fc0.a_raw).item():.6f}, b={softplus(fc0.b_raw).item():.6f}")

    elif f_type == "tanh":
        if "m_raw" in checkpoint:
            fc0.m_raw.data = torch.tensor(checkpoint["m_raw"], dtype=torch.float)

        m_list = checkpoint.get("m_list", [])

        print(f"Loaded m={softplus(fc0.m_raw).item():.6f}")

    else:
        alpha_list = beta_list = m_list = []
        #######
    print(f"Model loaded from {PATH}. f_type={f_type}")##新增
    return fc0
    






    

# TRAIN THE NETWORK.
# BY DEFAULT sin() activation and f=1-exp(-t) parametrization are used. These can
# To change them go to the class for network architecture and to the parametricSolution()

t0, t_max, N = 0., 20.*np.pi, 500;
dt = t_max/N;
# Set the initial state. lam controls the nonlinearity
x0, px0,  lam =  1.3, 1., 1
X0 = [t0, x0, px0, lam]
n_train, neurons, epochs, lr = N, 80, int(1e5), 5e-3

# # Training
# model_exp, loss_exp, loss_ham_exp, loss_energy_exp, _, alpha_exp, beta_exp, m_exp = trainModel(
#     X0, t_max, neurons, epochs, n_train, lr,
#     PATH="models/model_NL_exp.pt",
#     loadWeights=True,
#     f_type='exp', minLoss=1e-8
# )
#

# model_tanh, loss_tanh, loss_ham_tanh, loss_energy_tanh, _, alpha_tanh, beta_tanh, m_tanh = trainModel(
#     X0, t_max, neurons, epochs, n_train, lr,
#     PATH="models/model_NL_tanh.pt",
#     loadWeights=True,
#     f_type='tanh', minLoss=1e-8
# )

# model_v2, loss_v2, loss_ham_v2, loss_energy_v2, _, alpha_v2, beta_v2, m_v2 = trainModel(
#     X0, t_max, neurons, epochs, n_train, lr,
#     PATH="models/model_NL_v2.pt",
#     loadWeights=True,
#     f_type='v2', minLoss=1e-8
# )

# loadModel
model_exp = loadModel(PATH="models/model_NL_exp.pt", neurons=80, f_type='exp')

model_v2 = loadModel(PATH="models/model_NL_v2.pt", neurons=80, f_type='v2')

model_tanh = loadModel(PATH="models/model_NL_tanh.pt", neurons=80, f_type='tanh')


#####################################
# TEST THE PREDICTED SOLUTIONS
#######################################3
# nTest = N ; tTest = torch.linspace(t0,t_max,nTest)

nTest = 10*N ; t_max_test = 1.0*t_max
tTest = torch.linspace(t0,t_max_test,nTest)
tTest = tTest.reshape(-1,1);
tTest.requires_grad=True
t_net = tTest.detach().numpy().astype(np.float64)


x_exp, px_exp = parametricSolutions(tTest, model_exp, X0, f_type='exp')
x_v2, px_v2 = parametricSolutions(tTest, model_v2, X0, f_type='v2')
x_tanh, px_tanh = parametricSolutions(tTest, model_tanh, X0, f_type='tanh')

x_exp, px_exp = x_exp.data.numpy().astype(np.float64), px_exp.data.numpy().astype(np.float64)
x_v2, px_v2 = x_v2.data.numpy(), px_v2.data.numpy()
x_tanh, px_tanh = x_tanh.data.numpy(), px_tanh.data.numpy()

E_exp = energy(x_exp, px_exp, lam)
E_v2 = energy(x_v2, px_v2, lam)
E_tanh = energy(x_tanh, px_tanh, lam)

# ####################
# Scipy solver
######################
t_num = np.linspace(t0, t_max_test, nTest)
E0, E_ex = NLosc_exact(nTest,x0, px0, lam)
x_num,  px_num  = NLosc_solution(nTest ,t_num, x0,  px0,  lam)
E_num = energy( x_num,  px_num, lam)

# ###################
# # Symplectic Euler
# ####################
Ns = n_train -1; 
E_s, x_s, p_s, t_s = symEuler(Ns, x0, px0, t0, t_max_test,lam)
# 100 times more time points

T0 = time.time()
Ns100 = 100*n_train ;
E_s100, x_s100, p_s100, t_s100 = symEuler(Ns100, x0,px0,t0,t_max_test,lam)
runTimeEuler = time.time() -T0
print('Euler runtime is ',runTimeEuler/60 )



################
# Make the plots

lineW = 2 # Line thickness

plt.figure(figsize=(10,8))


plt.subplot(2,2,1)
plt.plot(t_num, x_num, '-g', linewidth=lineW, label='Ground truth')
# plt.plot(t_s, x_s, ':k', linewidth=lineW, label='Symplectic Euler')
# plt.plot(t_s100, x_s100, '-.k', linewidth=lineW, label='Symplectic Euler ×100')
plt.plot(t_net, x_exp, '--b', linewidth=lineW, label=r'$NN \  f(t)=1-e^{-t}$')
plt.plot(t_net, x_v2, '-.r', linewidth=lineW, label=r'$NN \  f(t)=\frac{1-e^{-\alpha t}}{1+\beta e^{-\alpha t}}$' )
plt.plot(t_net, x_tanh, '-.m', linewidth=lineW, label=r'$NN \ f(t)=\tanh(mt)$')

plt.ylabel('x'); plt.xlabel('t')
# plt.legend()

# ======================
# Energy comparison
# ======================
plt.subplot(2,2,2)
plt.plot(t_num, E_ex, '-g', linewidth=lineW, label='Ground truth')
# plt.plot(t_s, E_s, ':k', linewidth=lineW, label='Symplectic Euler')
# plt.plot(t_s100, E_s100, '-.k', linewidth=lineW, label='Symplectic Euler ×100')
plt.plot(t_net, E_exp, '--b', linewidth=lineW, label=r'$NN \  f(t)=1-e^{-t}$')
plt.plot(t_net, E_v2, '-.r', linewidth=lineW, label=r'$NN \  f(t)=\frac{1-e^{-\alpha t}}{1+\beta e^{-\alpha t}}$' )
plt.plot(t_net, E_tanh, '-.m', linewidth=lineW, label=r'$NN \ f(t)=\tanh(mt)$')

plt.ylabel('E'); plt.xlabel('t')
plt.ylim([1.95, 2.15])
plt.legend()

plt.subplot(2,2,3)
plt.plot(t_num, px_num, '-g', linewidth=lineW, label='Ground truth')
# plt.plot(t_s, p_s, ':k', linewidth=lineW, label='Symplectic Euler')
# plt.plot(t_s100, p_s100, '-.k', linewidth=lineW, label='Symplectic Euler ×100')
plt.plot(t_net, px_exp, '--b', linewidth=lineW, label=r'$NN \  f(t)=1-e^{-t}$')
plt.plot(t_net, px_v2, '-.r', linewidth=lineW, label=r'$NN \  f(t)=\frac{1-e^{-\alpha t}}{1+\beta e^{-\alpha t}}$')
plt.plot(t_net, px_tanh, '-.m', linewidth=lineW, label=r'$NN \ f(t)=\tanh(mt)$')

plt.ylabel('px'); plt.xlabel('t')
# plt.legend()

plt.subplot(2,2,4)
plt.plot(x_num, px_num, '-g', linewidth=lineW, label='Ground truth')
# plt.plot(x_s, p_s, ':k', linewidth=lineW, label='Symplectic Euler')
# plt.plot(x_s100, p_s100, '-.k', linewidth=lineW, label='Symplectic Euler ×100')
plt.plot(x_exp, px_exp, '--b', linewidth=lineW, label=r'$NN \  f(t)=1-e^{-t}$')
plt.plot(x_v2, px_v2, '-.r', linewidth=lineW, label=r'$NN \  f(t)=\frac{1-e^{-\alpha t}}{1+\beta e^{-\alpha t}}$' )
plt.plot(x_tanh, px_tanh, '-.m', linewidth=lineW, label=r'$NN \ f(t)=\tanh(mt)$')

plt.ylabel('p'); plt.xlabel('x')
# plt.legend()

# plt.savefig('nonlinearOscillator_trajectories.png')

plt.savefig('plots/nonlinearOscillator_trajectories_compare.pdf')



#####新增

# calculate the errors for networks
dx_exp = x_num - x_exp[:,0]; dp_exp = px_num - px_exp[:,0]
dx_v2 = x_num - x_v2[:,0]; dp_v2 = px_num - px_v2[:,0]
dx_tanh = x_num - x_tanh[:,0]; dp_tanh = px_num - px_tanh[:,0]

#################
# # calculate the errors for the solutions obtained by Euler
x_numN,  px_numN  = NLosc_solution(N,t_s, x0,  px0,  lam)
dx_s = x_numN - x_s;        dp_s = px_numN - p_s

x_num100,  px_num100  = NLosc_solution(N,t_s100, x0,  px0,  lam)
dx_s100 = x_num100 - x_s100;  dp_s100 = px_num100 - p_s100


plt.figure(figsize=(10,8))
# subplot (1,1): phase error scatter delta_p vs delta_x
plt.subplot(2,2,1)
# ground-truth point cloud (zero)
# plt.plot(dx_s, dp_s, ':k', linewidth=lineW, label='Symplectic Euler')
# plt.plot(dx_s100, dp_s100, '-.k', linewidth=lineW, label='Symplectic Euler ×100')
plt.plot(dx_exp, dp_exp, '--b', linewidth=lineW, label=r'$NN \  f(t)=1-e^{-t}$')
plt.plot(dx_v2, dp_v2, '-.r', linewidth=lineW, label=r'$NN \  f(t)=\frac{1-e^{-\alpha t}}{1+\beta e^{-\alpha t}}$')
plt.plot(dx_tanh, dp_tanh, '-.m', linewidth=lineW, label=r'$NN \ f(t)=\tanh(mt)$')
plt.ylim([-0.04, 0.04]); plt.xlim([-0.02, 0.02])
plt.ylabel(r'$\delta_p$'); plt.xlabel(r'$\delta_x$')
# plt.ylim([-6e-2, 6e-2]); plt.xlim([-6e-2, 6e-2])
# plt.legend()

# subplot (1,2): Energy zoom
plt.subplot(2,2,2)
plt.plot(t_num, E_ex, '-g', linewidth=lineW, label='Ground truth')
# plt.plot(t_s, E_s, ':k', linewidth=lineW, label='Symplectic Euler')
# plt.plot(t_s100, E_s100, '-.k', linewidth=lineW, label='Symplectic Euler ×100')
plt.plot(t_net, E_exp, '--b', linewidth=lineW, label=r'$NN \  f(t)=1-e^{-t}$')
plt.plot(t_net, E_v2, '-.r', linewidth=lineW, label=r'$NN \  f(t)=\frac{1-e^{-\alpha t}}{1+\beta e^{-\alpha t}}$')
plt.plot(t_net, E_tanh, '-.m', linewidth=lineW, label=r'$NN \ f(t)=\tanh(mt)$')

plt.ylabel('E'); plt.xlabel('t')
plt.ylim([1.95,2.15])
plt.legend()

# subplot (2,1): delta_x (time series)
plt.subplot(2,2,3)
# plt.plot(t_s, dx_s, ':k', linewidth=lineW, label='Symplectic Euler')
# plt.plot(t_s100, dx_s100, '-.k', linewidth=lineW, label='Symplectic Euler ×100')
plt.plot(t_num, dx_exp, '--b', linewidth=lineW, label=r'$NN \  f(t)=1-e^{-t}$')
plt.plot(t_num, dx_v2, '-.r', linewidth=lineW, label=r'$NN \  f(t)=\frac{1-e^{-\alpha t}}{1+\beta e^{-\alpha t}}$')
plt.plot(t_num, dx_tanh, '-.m', linewidth=lineW, label=r'$NN \  f(t)=\tanh(mt)$')

plt.ylim([-0.05, 0.05])
plt.ylabel(r'$\delta_x$'); plt.xlabel('t')
# plt.legend()

# subplot (2,2): delta_p (time series)
plt.subplot(2,2,4)
# plt.plot(t_s, dp_s, ':k', linewidth=lineW, label='Symplectic Euler')
# plt.plot(t_s100, dp_s100, '-.k', linewidth=lineW, label='Symplectic Euler ×100')
plt.plot(t_num, dp_exp, '--b', linewidth=lineW, label=r'$NN \  f(t)=1-e^{-t}$')
plt.plot(t_num, dp_v2, '-.r', linewidth=lineW, label=r'$NN \  f(t)=\frac{1-e^{-\alpha t}}{1+\beta e^{-\alpha t}}$')
plt.plot(t_num, dp_tanh, '-.m', linewidth=lineW, label=r'$NN \  f(t)=\tanh(mt)$')
plt.ylim([-0.06, 0.06])
plt.ylabel(r'$\delta_p$'); plt.xlabel('t')
# plt.legend()

plt.tight_layout()
plt.savefig('plots/nonlinearOscillator_error_compare.pdf')


# plt.show()
paths = [
    "models/model_NL_exp_loss.txt",
    "models/model_NL_v2_loss.txt",
    "models/model_NL_tanh_loss.txt"
]
labels = [
    r'$f(t)=1-e^{-t}$',
    r'$f(t)=\frac{1-e^{-\alpha t}}{1+\beta e^{-\alpha t}}$',
    r'$f(t)=\tanh(mt)$'
]
styles = [
    "-b",
     "-r",
          "-m"
          ]

plt.figure(figsize=(6,5))
for p, label, style in zip(paths, labels, styles):
    if os.path.exists(p):
        loss = np.loadtxt(p)
        plt.semilogy(loss, style, linewidth=2, label=label)
    else:
        print(f"⚠️ Warning: {p} not found, skip.")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.grid(True, which="both", ls=":")
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/loss_comparison.pdf")

x_exp = np.array(x_exp)
px_exp = np.array(px_exp)
x_v2 = np.array(x_v2)
px_v2 = np.array(px_v2)
x_tanh = np.array(x_tanh)
px_tanh = np.array(px_tanh)

# === 误差计算 ===
# L2范数误差
L2_x_exp = np.linalg.norm(x_exp - x_num.reshape(-1,1))
L2_x_v2 = np.linalg.norm(x_v2 - x_num.reshape(-1,1))
L2_px_exp = np.linalg.norm(px_exp - px_num.reshape(-1,1))
L2_px_v2 = np.linalg.norm(px_v2 - px_num.reshape(-1,1))
L2_x_tanh = np.linalg.norm(x_tanh - x_num.reshape(-1,1))
L2_px_tanh = np.linalg.norm(px_tanh - px_num.reshape(-1,1))

# MSE误差
MSE_x_exp = mean_squared_error(x_num, x_exp)
MSE_x_v2 = mean_squared_error(x_num, x_v2)
MSE_px_exp = mean_squared_error(px_num, px_exp)
MSE_px_v2 = mean_squared_error(px_num, px_v2)
MSE_x_tanh = mean_squared_error(x_num, x_tanh)
MSE_px_tanh = mean_squared_error(px_num, px_tanh)

# === Euler 误差 ===
L2_x_euler  = np.linalg.norm(dx_s)      # x 的 L2
L2_px_euler = np.linalg.norm(dp_s)      # p 的 L2

MSE_x_euler  = np.mean(dx_s**2)
MSE_px_euler = np.mean(dp_s**2)
print("\n=================== ERROR COMPARISON ===================")
print(f"L2(x): Euler={L2_x_euler:.4e}, exp={L2_x_exp:.4e}, tanh={L2_x_tanh:.4e}, v2={L2_x_v2:.4e}")
print(f"L2(px): Euler={L2_px_euler:.4e}, exp={L2_px_exp:.4e}, tanh={L2_px_tanh:.4e}, v2={L2_px_v2:.4e}")

print(f"MSE(x): Euler={MSE_x_euler:.4e}, exp={MSE_x_exp:.4e}, tanh={MSE_x_tanh:.4e}, v2={MSE_x_v2:.4e}")
print(f"MSE(px): Euler={MSE_px_euler:.4e}, exp={MSE_px_exp:.4e}, tanh={MSE_px_tanh:.4e}, v2={MSE_px_v2:.4e}")
print("========================================================\n")
# ================================
# SAVE ERROR RESULTS TO TXT FILE
# ================================
os.makedirs("data", exist_ok=True)

error_table = np.array([
    ["Model", "L2(x)", "L2(px)", "MSE(x)", "MSE(px)"],

    ["Euler", L2_x_euler, L2_px_euler, MSE_x_euler, MSE_px_euler],
    ["exp",   L2_x_exp,   L2_px_exp,    MSE_x_exp,   MSE_px_exp],
    ["tanh",  L2_x_tanh,  L2_px_tanh,   MSE_x_tanh,  MSE_px_tanh],
    ["v2",    L2_x_v2,    L2_px_v2,     MSE_x_v2,    MSE_px_v2]
], dtype=object)

np.savetxt("data/error_comparison.txt",
           error_table,
           fmt="%s",
           delimiter="\t")

print(" Error table saved to: data/error_comparison.txt")

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(t_num, np.abs(x_exp[:,0] - x_num), '-b', label='exp')
plt.plot(t_num, np.abs(x_v2[:,0] - x_num), '-r', label='v2')
plt.plot(t_num, np.abs(x_tanh[:,0] - x_num), '-m', label='tanh')
plt.xlabel("t"); plt.ylabel("|Δx|")
plt.title("Position absolute error")
# plt.legend();
plt.grid(True, ls=':')

plt.subplot(1,2,2)
plt.plot(t_num, np.abs(px_exp[:,0] - px_num), '-b', label='exp')
plt.plot(t_num, np.abs(px_v2[:,0] - px_num), '-r', label='v2')
plt.plot(t_num, np.abs(px_tanh[:,0] - px_num), '-m', label='tanh')
plt.xlabel("t"); plt.ylabel("|Δp|")
plt.title("Momentum absolute error")
# plt.legend();
plt.grid(True, ls=':')

plt.tight_layout()
plt.savefig("plots/abs_error.pdf")


#
# # # === plot parameter evolution for v2 ===新增
# if 'alpha_v2' in locals() and 'beta_v2' in locals():
#     plt.figure()
#     plt.plot(alpha_v2, label='alpha')
#     plt.plot(beta_v2, label='beta')
#     plt.xlabel("Epoch")
#     plt.ylabel("Value")
#     plt.title("Evolution of alpha & beta")
#     plt.legend()
#     plt.savefig("plots/param_evolution_v2.pdf")
# if 'm_tanh' in locals():
#     plt.figure()
#     plt.plot(m_tanh, label='m')
#     plt.xlabel("Epoch")
#     plt.ylabel("Value")
#     plt.title("Evolution of m")
#     plt.legend()
#     plt.savefig("plots/param_evolution_tanh.pdf")


paths = [
    ("models/model_NL_exp",   r"$f(t)=1-e^{-t}$"),
    ("models/model_NL_tanh",  r"$f(t)=\tanh(mt)$"),
    ("models/model_NL_v2",    r"$f(t)=\frac{1-e^{-\alpha t}}{1+\beta e^{-\alpha t}}$")
]

plt.figure(figsize=(6,5))
#
# for base, label in paths:
#     loss_ham = np.loadtxt(base + "_loss_ham.txt")
#     loss_energy = np.loadtxt(base + "_loss_energy.txt")
#
#     plt.semilogy(loss_ham, '--', label=label + " (Ham)")
#     plt.semilogy(loss_energy, '-', label=label + " (Energy)")
#
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.grid(True, which="both", ls=":")
# plt.tight_layout()
# plt.savefig("plots/loss_split.pdf")

# plt.show()

# saveData('data/', t_net, x, px, E)
# saveData('data/Euler100/', t_s100, x_s100, p_s100, E_s100)


# np.savetxt('data/'+"dx.txt",dx)
# np.savetxt('data/'+"dp.txt",dp)
# np.savetxt('data/Euler100/'+"dx.txt",dx_s100)
# np.savetxt('data/Euler100/'+"dp.txt",dp_s100)

plt.show()

mat_data = {
    # time grids
    "t_net": t_net,                  # NN time grid
    "t_num": t_num,                  # ground truth time grid
    "t_s": t_s,                      # symplectic Euler
    "t_s100": t_s100,                # symplectic Euler x100

    # ground truth solutions
    "x_num": x_num,
    "px_num": px_num,
    "E_ex": E_ex,

    # NN predictions
    "x_exp": x_exp,
    "px_exp": px_exp,
    "E_exp": E_exp,

    "x_tanh": x_tanh,
    "px_tanh": px_tanh,
    "E_tanh": E_tanh,

    "x_v2": x_v2,
    "px_v2": px_v2,
    "E_v2": E_v2,

    # symplectic Euler
    "x_s": x_s,
    "p_s": p_s,
    "E_s": E_s,

    "x_s100": x_s100,
    "p_s100": p_s100,
    "E_s100": E_s100,

    # errors (time series)
    "dx_exp": dx_exp,
    "dp_exp": dp_exp,
    "dx_tanh": dx_tanh,
    "dp_tanh": dp_tanh,
    "dx_v2": dx_v2,
    "dp_v2": dp_v2,

    "dx_s": dx_s,
    "dp_s": dp_s,
    "dx_s100": dx_s100,
    "dp_s100": dp_s100,

    # scalar error metrics
    "L2_x_exp": L2_x_exp,
    "L2_px_exp": L2_px_exp,
    "L2_x_tanh": L2_x_tanh,
    "L2_px_tanh": L2_px_tanh,
    "L2_x_v2": L2_x_v2,
    "L2_px_v2": L2_px_v2,

    "MSE_x_exp": MSE_x_exp,
    "MSE_px_exp": MSE_px_exp,
    "MSE_x_tanh": MSE_x_tanh,
    "MSE_px_tanh": MSE_px_tanh,
    "MSE_x_v2": MSE_x_v2,
    "MSE_px_v2": MSE_px_v2,

    "L2_x_euler": L2_x_euler,
    "L2_px_euler": L2_px_euler,
    "MSE_x_euler": MSE_x_euler,
    "MSE_px_euler": MSE_px_euler,
}
# mat_data = {
#     # NN predictions
#     "x_base": x_exp,
#     "px_base": px_exp,
#     "E_base": E_exp,
#     # errors (time series)
#     "dx_base": dx_exp,
#     "dp_base": dp_exp,
#     # scalar error metrics
#     "L2_x_base": L2_x_exp,
#     "L2_px_base": L2_px_exp,
#     "MSE_x_base": MSE_x_exp,
#     "MSE_px_base": MSE_px_exp,
# }
os.makedirs("results", exist_ok=True)
# savemat("results/NL_transfer_base.mat", mat_data)
savemat("results/NL_transfer.mat", mat_data)
print("Saved")
