"quantum circuits"
"task 3"
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import scipy
import scipy.optimize
import sys
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import axes3d
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QHBoxLayout,
                             QLabel, QSlider, QVBoxLayout, QWidget)
from PyQt6.QtGui import QAction

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

"def C and D"
pi = np.pi
tstep = 100
t_seq = np.linspace(-1, 1, tstep)
# ga_seq = np.linspace(0, np.pi / 2, 100)
C = get_U(0, 0)
D = get_U(np.pi, 0)
Ua = D
Ub = Q  
grid = []
for x, tA in enumerate(t_seq):
    for y, tB in enumerate(t_seq):
        if tA >=0:
            Ua = get_U(tA*pi, 0)
        else:
            Ua = get_U(0, tA*pi/2)
        if tB >=0:
            Ub = get_U(tB*pi, 0)
        else:
            Ub = get_U(0, tB*pi/2)
        grid.append([Ua, Ub])
grid = np.array(grid)
grid.shape = tstep, tstep, 2, 2, 2
# ga = np.pi/2
ga = 0
val_map = np.zeros((tstep, tstep))
for x in range(tstep):
    for y in range(tstep):
        out = get_sequencial_transform(grid[x, y, 0], grid[x, y, 1], D, ga)
        val = get_payoffs_A(out)
        assert(np.allclose(np.imag(val), 0))
        val = np.real(val)
        val_map[x, y] = val


z = val_map.T
x, y = np.meshgrid(t_seq, t_seq)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(x, y, z, cmap='viridis')

# # Set labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Surface plot of Z(x, y)')
from plot3D import ApplicationWindow
app = QApplication(sys.argv)
w = ApplicationWindow(x, y, z)
w.setFixedSize(1280, 720)
# w.plot_surface()
w.show()
sys.exit(app.exec())
