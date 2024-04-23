"quantum circuits"
"task 2"
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import scipy
import scipy.optimize
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{physics}')
# %%
"operators definition"
I = np.eye(2)
D = np.array([[0, 1], [-1, 0]])
Q = np.array([[1j, 0], [0, -1j]])
M = 1 / np.sqrt(2) * np.array([[1j, -1], [1, -1j]])
c = np.cos
s = np.sin
weights_A = np.array([3, 0, 5, 1])
weights_B = np.array([3, 5, 0, 1])
ket_0 = np.array([1, 0])
ket_1 = np.array([0, 1])
states = np.array([0, 1])


def get_J(D, ga):
    assert ga >= 0 and ga <= np.pi / 2
    prod = np.kron(D, D)
    op = expm(-1j * ga * prod / 2)
    op_cj = expm(1j * ga * prod / 2)
    return op, op_cj


def get_U(th, phi):
    M = np.array(
        [
            [np.exp(1j * phi) * c(th / 2), s(th / 2)],
            [-s(th / 2), np.exp(-1j * phi) * c(th / 2)],
        ]
    )
    return M


def get_payoffs_A(psi):
    vals = psi * psi.conj()
    assert len(vals) == 4
    pay = np.sum(weights_A * vals)
    return pay


def get_payoffs_B(psi):
    vals = psi * psi.conj()
    assert len(vals) == 4
    pay = np.sum(weights_B * vals)
    return pay


def get_sequencial_transform(Ua, Ub, D, ga):
    J, Jcj = get_J(D, ga)
    U = np.kron(Ua, Ub)
    tot = Jcj @ U @ J
    psi_in = np.kron(ket_0, ket_0)
    out = tot @ psi_in
    return out


# %%
"def C and D"
ga_seq = np.linspace(0, np.pi / 2, 100)
C = get_U(0, 0)
D = get_U(np.pi, 0)
Ua = D
Ub = Q
param_seq = [(D, D), (D, Q), (Q, D), (Q, Q)]
name_seq = ["(D,D)", "(D, Q)", "(Q, D)", "(Q, Q)"]
for i, param in enumerate(param_seq):
    Ua, Ub = param
    val_seq = []
    for ga in ga_seq:
        out = get_sequencial_transform(Ua, Ub, D, ga)
        val = get_payoffs_A(out)
        val_seq.append(val)
    plt.plot(ga_seq, val_seq, label=name_seq[i])
    plt.ylabel('Alice payoff')
    plt.xlabel(r'$\gamma$')
    plt.legend()
plt.savefig('plot1.png', dpi = 600)
# %%
"dla 2b"
"maksymalne gamma = pi/2"
ga = np.pi / 2
param_seq = [C, D, M]
th_seq = np.linspace(0, 2 * np.pi, 100)
name_seq = ["(C, U)", "(D, U)", "(M, U)"]
for i, param in enumerate(param_seq):
    Ua = param
    val_seq = []
    for th in th_seq:
        Ub = get_U(th, 0)
        out = get_sequencial_transform(Ua, Ub, D, ga)
        val = get_payoffs_A(out)
        val_seq.append(val)
    plt.plot(th_seq, val_seq, label=name_seq[i])
    plt.xlabel(r"$\theta$")
    plt.ylabel('Alice payoff')
    plt.legend()
plt.savefig('plot2.png', dpi = 600)
# %%
"ptk c"


def payoff_A(angles, *args):
    ga = args[0]
    theta, phi = angles
    Ua = get_U(theta, phi)
    Ub = C
    out = get_sequencial_transform(Ua, Ub, D, ga)
    payoff = get_payoffs_A(out)
    return -payoff  # diff. evo. minimizes, so to maximize we minimize the negation


def opt(ga):
    result = scipy.optimize.differential_evolution(
        payoff_A, args=(ga,), bounds=[(0, np.pi), (0, np.pi / 2)]
    )
    return result.fun


ga_seq = np.linspace(0, np.pi / 2, 100)
res = []
for ga in ga_seq:
    res.append(opt(ga))
res2 = np.array(res)

# %%
plt.plot(ga_seq, -res1, label = r'$\hat{U}_B = \hat{D}$')
# plt.plot(ga_seq, -res2, label = r'$\hat{U}_B = \hat{C}$')
plt.legend()
plt.ylabel('Alice payoff')
# plt.title(r'$\hat{U}_B = \hat{C}$')
plt.xlabel(r'$\gamma$')
plt.savefig('plot3.png', dpi = 600)
# %%
