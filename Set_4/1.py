'hydrodynamics'
#%%
import numpy as np
import matplotlib.pyplot as plt
# %%
'initial parameters'
_ = np.newaxis
direction = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
Nx = 520
Ny = 180
u_in = 0.04
Re = 220
vidx = 9
visc = u_in*(Ny/2)/Re
tau = 3*visc + 1/2
eps = 0.0001
rho0 = 1
tau = 3*visc + 1/2
# %%
'create space'
'1 - obstascle'
space = np.fromfunction(lambda x,y: abs(x-Nx/4)+abs(y)<Ny/2, shape = (Nx, Ny))*1.0
space[0, :Ny] = 1
space[:Nx, 0] = 1
f_lattice = np.zeros((Nx, Ny, 9))
v_lattice = np.zeros((Nx, Ny, 2))
rho_lattice = np.zeros((Nx, Ny))
f_eq = np.zeros((Nx, Ny))
'init velosicty'
'ex = dir[1]'
y = np.linspace(0, Ny, Ny)
u0 = u_in*(1+eps*np.sin(2*np.pi*y/(Ny-1)))
f_lattice[:,:,1] = u0

def D2norm(mat):
    temp = np.copy(mat)
    return(np.sum(temp*temp, axis = 2))

def calc_feq(rho_lattice, f_lattice, v_lattice, init = False):
    prod = np.sum(direction[_,_,:,:]*v_lattice[:,:,_,:], axis = 3)
    if init:
        rho_lattice[:,:] = rho0
    f_eq = W[_,_,:]*rho_lattice[:,:, _]*(1 + 3*prod[:,:,:]+9/2*(prod[:,:,:])**2-3/2*D2norm(v_lattice)[:,:,_])
    return f_eq

def calc_inlet_outlet(rho_lattice, f_lattice, f_eq):
    f = f_lattice
    data = (2*(f[0,:,3]+f[0,:,6]+f[0,:,7])+f[0,:,0]+f[0,:,2]+f[0,:,4])/(1-np.abs(u0))
    rho_lattice[0,:] = data
    f[0,:, 1] = f_eq[0,:, 1]
    f[0,:, 5] = f_eq[0,:, 5]
    f[0,:, 8] = f_eq[0,:, 8]
    f[Nx-1, :, 3] = f[Nx-2, :, 3]
    f[Nx-1, :, 6] = f[Nx-2, :, 6]
    f[Nx-1, :, 7] = f[Nx-2, :, 7]
    return f
def calc_density(f_lattice):
    rho = np.sum(f_lattice[:,:,:], axis = 2)
    return rho

def calc_velosity(rho_lattice, f_lattice):
    rho = 1/rho_lattice[:,:,_]*np.sum(f_lattice[:,:,:,_]*direction[_,_,:,:], axis = 2)
    return rho
#%%
'test initial'
f_init = calc_feq(rho_lattice, f_lattice, v_lattice, init=True)
f_lattice = np.copy(f_init)
f_lattice = calc_inlet_outlet(rho_lattice, f_lattice, f_init)
rho_lattice = calc_density(f_lattice)
v_lattice = calc_velosity(rho_lattice, f_lattice)
f_eq =  calc_feq(rho_lattice, f_lattice, v_lattice)
f_col = f_lattice-(f_lattice-f_eq)/tau

#%%
plotek = D2norm(v_lattice)
plt.imshow(plotek.T)
plt.colorbar()
# %%
