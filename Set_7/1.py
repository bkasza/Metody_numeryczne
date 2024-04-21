"quantum circuits"
"task 1"
# %%
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{physics}')


# %%
c = np.cos
s = np.sin
states = np.array([0, 1])


def get_U(al):
    M = np.array([[c(al / 2), s(al / 2)], [-s(al / 2), c(al / 2)]])
    return M

def measure(psi):
    prob_seq = psi * psi.conj()
    out = []
    print(prob_seq.shape)
    for p in prob_seq:
        result = np.random.choice(states, size = 1, p = p)
        out.append(result)
    out = np.array(out)
    return out


def get_random_states(N=100):
    out = []
    for i in range(N):
        al = np.random.random() * 2 * np.pi
        out.append(np.array([c(al / 2), s(al / 2)]))
    out = np.array(out)
    return out

def get_states(N, val = None):
    if val is not None:
        al = val
    else:
        al = float(input())
    U = get_U(al)
    psi = U @ np.array([1,0])
    psi_seq = np.tile(psi, N)
    psi_seq.shape = N, 2
    return psi_seq, al
# %%
N = 1000
state_seq, al = get_states(N, np.pi/2)
# state_seq = get_random_states()
prob = (np.cos(al/2)**2, np.sin(al/2)**2)
meas_seq = measure(state_seq)
meas_seq = meas_seq
plt.hist(meas_seq, bins = 3)
plt.ylabel('Counts')
plt.title(rf'Measurement vs expected values: $p(\ket{0}) = $ {prob[0].round((2))}, $p(\ket{1}) = $ {prob[1].round((2))}')
# plt.xticks([1/6, 5/6], [r'$\ket{0}$', r'$\ket{1}$'])
# %%
