#%%
'analog quantum compupting'
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
# %%
h_seq = np.array([-0.5, 0.5, -0.1])
J_seq = np.array([-0.4, -1.6, -1.0])
n = 2
pX, pY, pZ = np.array([[0,1],[1,0]]),np.array([[0,1j],[-1j,0]]), np.array([[1,0],[0,-1]])
I = np.eye(n)

# %%
def get_tripple_cron(a, b, c):
    return np.kron(np.kron(a, b), c) 
SZ_seq = [
    get_tripple_cron(pZ, I, I),
    get_tripple_cron(I, pZ, I),
    get_tripple_cron(I, I, pZ)
]
SX_seq = [
    get_tripple_cron(pX, I, I),
    get_tripple_cron(I, pX, I),
    get_tripple_cron(I, I, pX)
]

def get_H(la = 0):
    H0 = np.zeros((n**3, n**3))
    for S in SX_seq:
        H0 += -S
    H1 = -J_seq[0]*SZ_seq[0]@SZ_seq[1]-J_seq[1]*SZ_seq[0]@SZ_seq[2]-J_seq[2]*SZ_seq[1]@SZ_seq[2]
    H1 = H1 - h_seq[0]*SZ_seq[0] - h_seq[1]*SZ_seq[1] - h_seq[2]*SZ_seq[2]
    H = (1-la)*(H0)+la*H1
    return H

def evol(psi0, steps, Tm):
    dt = Tm/steps 
    psi = psi0.copy()
    exp_seq = []
    for k in range(steps):
        tk = k
        la = tk/steps
        H = get_H(la)
        U = sc.linalg.expm(-1j*H*dt)
        psi = U @ psi
        val = []
        for S in SZ_seq:
            val.append(psi.T.conj()@S@psi)
        exp_seq.append(val)
    return exp_seq

def evol_fidelity(psi0, psif, steps, Tm):
    dt = Tm/steps 
    psi = psi0.copy()
    fid_seq = []
    for k in range(steps):
        tk = k
        la = tk/steps
        H = get_H(la)
        U = sc.linalg.expm(-1j*H*dt)
        psi = U @ psi
        fid_seq.append(abs(psif.T@psi)**2)
    return fid_seq
#%%
'init democratic eigenstate'
v = np.array([1,1])/np.sqrt(2) 
Tm = 1000
psif = np.array([0, 1, 0, 0, 0, 0, 0, 0])
steps = 10000
psi0 = get_tripple_cron(v, v, v)
exp_seq = evol(psi0, steps, Tm)
fid_seq = evol_fidelity(psi0, psif, steps, Tm)
# %%
plt.plot(np.linspace(0, 1, steps), exp_seq, label = ['S0', 'S1', 'S2'])
plt.xlabel('tstep')
plt.ylabel('expval')
plt.legend()
plt.title(f'TM:{Tm}, steps:{steps}')
# %%
plt.plot(np.linspace(0, 1, steps), fid_seq)
plt.title(f'TM:{Tm}, steps:{steps}')

# %%
