"quantum circuits"
"task 1"
# %%
import numpy as np
import matplotlib.pyplot as plt

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

def get_states(N):
    al = float(input())
    U = get_U(al)
    psi = U @ np.array([1,0])
    psi_seq = np.tile(psi, N)
    psi_seq.shape = N, 2
    return psi_seq
# %%
state_seq = get_states(20)
# state_seq = get_random_states()
meas_seq = measure(state_seq)
plt.hist(meas_seq, range=[0, 1])
# %%
