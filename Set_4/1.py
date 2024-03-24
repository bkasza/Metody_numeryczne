"hydrodynamics"
# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
"initial parameters"
_ = np.newaxis
direction = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
)
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
Nx = 520
Ny = 180
u_in = 0.04
Re = 220
vidx = 9
visc = u_in * (Ny / 2) / Re
tau = 3 * visc + 1 / 2
eps = 0.0001
rho0 = 1
tau = 3 * visc + 1 / 2
new_idx = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
# %%
"create space"
"1 - obstascle"
space = (
    np.fromfunction(lambda x, y: abs(x - Nx / 4) + abs(y) < Ny / 2, shape=(Nx, Ny))
    * 1.0
)
space[0, :Ny] = 1
space[:Nx, 0] = 1

obst_idx = np.where(space == 1)


def D2norm(mat):
    return np.sum(mat * mat, axis=2)


def calc_feq_0(rho_lattice, f_lattice, v_lattice, init=False):
    "step 1.1"
    prod = np.sum(direction[_, _, :, :] * v_lattice[:, :, _, :], axis=3)
    if init:
        rho_lattice[:, :] = rho0
    f = (
        W[_, _, :]
        * rho_lattice[0, :, _]
        * (
            1
            + 3 * prod[0, :, :]
            + 9 / 2 * (prod[0, :, :]) ** 2
            - 3 / 2 * D2norm(v_lattice)[0, :, _]
        )
    )
    f_lattice[0, :, :] = f
    return f_lattice


def calc_rho0(rho_lattice, f_lattice, f_eq):
    "step 1.2"
    f = f_lattice.copy()
    data = (
        2 * (f[0, :, 3] + f[0, :, 6] + f[0, :, 7])
        + f[0, :, 0]
        + f[0, :, 2]
        + f[0, :, 4]
    ) / (1 - np.abs(u0))
    rho_lattice[0, :] = data
    return rho_lattice

def calc_inlet_outlet(rho_lattice, f_lattice, f_eq):
    f = f_lattice.copy()
    f[0, :, 1] = f_eq[0, :, 1]
    f[0, :, 5] = f_eq[0, :, 5]
    f[0, :, 8] = f_eq[0, :, 8]
    f[Nx - 1, :, 3] = f[Nx - 2, :, 3]
    f[Nx - 1, :, 6] = f[Nx - 2, :, 6]
    f[Nx - 1, :, 7] = f[Nx - 2, :, 7]
    return f


def calc_density(f_lattice):
    "step 3.1"
    rho = np.sum(f_lattice[:, :, :], axis=2)
    return rho


def calc_velosity(rho_lattice, f_lattice):
    "step 3.2"
    u = (
        1 / rho_lattice[:, :, _]
        * np.sum(f_lattice[:, :, :, _] * direction[_, _, :, :], axis=2)
    )
    return u


def calc_feq(rho_lattice, f_lattice, v_lattice):
    "step 3.3"
    prod = np.sum(direction[_, _, :, :] * v_lattice[:, :, _, :], axis=3)
    f_eq = (
        W[_, _, :]
        * rho_lattice[:, :, _]
        * (
            1
            + 3 * prod[:, :, :]
            + 9 / 2 * (prod[:, :, :]) ** 2
            - 3 / 2 * D2norm(v_lattice)[:, :, _]
        )
    )
    return f_eq


def calc_fcol(f_lattice, f_eq):
    "step 4,5"
    f = f_lattice - (f_lattice - f_eq) / tau
    f[obst_idx][:] = f_lattice[obst_idx][:, new_idx]
    return f


def calc_stream(f_lattice):
    "step 6"
    for i, d in enumerate(direction):
        f_lattice[:, :, i] = np.roll(f_lattice[:, :, i], d, axis=(0, 1))
    return f_lattice


# %%
"petlik"
f_lattice = np.zeros((Nx, Ny, 9))
v0_lattice = np.zeros((Nx, Ny, 2))
v_lattice = np.zeros((Nx, Ny, 2))
rho_lattice = np.ones((Nx, Ny))
f_eq = np.zeros((Nx, Ny))
"init velosicty"
"ex = dir[1]"
y = np.linspace(0, Ny, Ny)
u0 = u_in * (1 + eps * np.sin(2 * np.pi * y / (Ny - 1)))
v0_lattice[:, :, 0] = u0[_,:]

steps = 10
f_eq = calc_feq(rho_lattice, f_lattice, v0_lattice)
f_lattice = np.copy(f_eq)
for s in range(steps):
    f_eq0 = calc_feq_0(rho_lattice, f_lattice, v0_lattice)
    f_lattice = calc_inlet_outlet(rho_lattice, f_lattice, f_eq0)
    rho_lattice = calc_density(f_lattice)
    v_lattice = calc_velosity(rho_lattice, f_lattice)
    f_eq = calc_feq(rho_lattice, f_lattice, v_lattice)
    f_lattice = calc_fcol(f_lattice, f_eq)
    f_lattice = calc_stream(f_lattice)

# %%
plotek = D2norm(v_lattice)
plt.imshow(plotek.T, cmap="hot")
plt.colorbar()
# %%
