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


def calc_ground_state(la_seq):
    e_seq = []
    for la in la_seq:
        H = get_H(la=la)
        en, v = np.linalg.eigh(H)
        e_seq.append(en[0])
    return e_seq
la_seq = np.linspace(0, 1, 100) 
e_seq = calc_ground_state(la_seq)

def calc_exp_val(la_seq):
    exp_seq = []
    for la in la_seq:
        H = get_H(la)
        en, v = np.linalg.eigh(H)
        gstate = v[:, 0]
        val = []
        for S in SZ_seq:
            val.append(gstate.T@S@gstate)
        exp_seq.append(val)
    print(gstate)
    return np.array(exp_seq)
#%%
la_seq = np.linspace(0, 1, 100)
e_seq = calc_ground_state(la_seq)
exp_seq = calc_exp_val(la_seq)
# %%
plt.plot(la_seq, e_seq)
# %%
plt.plot(la_seq, exp_seq, label = ['0', '1', '2'])
plt.xlabel('lambda')
plt.ylabel('expval')
plt.legend()
# %%
