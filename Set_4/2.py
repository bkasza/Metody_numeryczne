"hydrodynamics"
'vol 2 cyllinder'
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
Re = 20
visc = u_in * (Ny / 2) / Re
tau = 3 * visc + 1 / 2
eps = 0.0001
rho0 = 1
tau = 3 * visc + 1 / 2
new_idx = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
# %%
"create space"
"1 - obstacle"
radius = 30
center_x = Nx // 2  # Positioning in 1/8 of x direction
center_y = Ny // 2   # Centered in y direction

space = np.fromfunction(lambda x, y: ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius ** 2, shape=(Nx, Ny), dtype=int)*1

obst_idx = np.where(space == 1)


def D2norm(mat):
    return np.sum(mat * mat, axis=2)

def calc_rho0(rho_lattice, f_lattice):
    "step 1.1"
    f = f_lattice.copy()
    data = (
        2 * (f[0, :, 3] + f[0, :, 6] + f[0, :, 7])
        + f[0, :, 0]
        + f[0, :, 2]
        + f[0, :, 4]
    ) / (1 - np.abs(u0))
    # ) / (1 - np.sqrt(np.sum(v0_lattice[0, :, :]*v0_lattice[0, :, :], axis=1)))
    rho_lattice[0, :] = data
    return rho_lattice

def calc_feq_0(rho_lattice, f_eq, v_lattice):
    "step 1.2"
    prod = np.sum(direction[_, _, :, :] * v_lattice[:, :, _, :], axis=3)
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
    f_eq[0, :, :] = f
    return f_eq

def calc_inlet_outlet(f_lattice, f_eq):
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


def calc_feq(rho_lattice, v_lattice):
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
    fcol = f_lattice - (f_lattice - f_eq) / tau
    f = np.zeros(f_lattice.shape)
    for i, idx in enumerate(new_idx):
        # fcol[obst_idx][i] = f_lattice[obst_idx][idx]
        f[:,:, i] = f_lattice[:,:,idx]
    fcol[obst_idx] = f[obst_idx]
    return fcol


def calc_stream(f_lattice):
    "step 6"
    f = np.zeros(f_lattice.shape)
    for i, d in enumerate(direction):
        f[:, :, i] = np.roll(f_lattice[:, :, i], d, axis=(0, 1))
    return f


# %%
"petlik"
f_lattice = np.zeros((Nx, Ny, 9))
v0_lattice = np.zeros((Nx, Ny, 2))
v_lattice = np.zeros((Nx, Ny, 2))
rho_lattice = np.ones((Nx, Ny))
f_eq = np.zeros((Nx, Ny))
"init velosicty"
"ex = dir[1]"
y = np.linspace(0, Ny-1, Ny)
u0 = u_in * (1 + eps * np.sin(2 * np.pi * y / (Ny - 1)))
v0_lattice[:, :, 0] = u0[_,:]
v_lattice = v0_lattice

steps = 10001
photo_step = 100
f_eq = calc_feq(rho_lattice, v0_lattice)
f_lattice = np.copy(f_eq)
for s in range(steps):
    rho_lattice = calc_rho0(rho_lattice, f_lattice)
    f_eq0 = calc_feq_0(rho_lattice, f_eq, v0_lattice)
    f_lattice = calc_inlet_outlet(f_lattice, f_eq0)
    rho_lattice = calc_density(f_lattice)
    v_lattice = calc_velosity(rho_lattice, f_lattice)
    f_eq = calc_feq(rho_lattice, v_lattice)
    f_col = calc_fcol(f_lattice, f_eq)
    f_lattice = calc_stream(f_col)
    if s%photo_step == 0:    
        plotek = D2norm(v_lattice)
        plt.gcf().set_facecolor("pink")
        plt.imshow(plotek.T, cmap="hot")    
        plt.title(f'r: {radius}, re: {Re}')
        plt.savefig(f'{s:06d}.png')
        plt.close()
# %%
plotek = D2norm(v_lattice)
plt.imshow(plotek.T, cmap="hot")
plt.colorbar()
# %%
print(f_lattice[:,:,1])
# %%