#%%
'monte carlo with numba, vanila step'
import numpy as np
import numba as nb
#%%
N = 8
be = 4
de = 1
x = np.linspace(-de, de, N)
Dt = be/N

# %%

@nb.jit(nopython = True)
def rho_free(x, y):
    return np.exp(-(x-y)**2/(2*Dt))

@nb.jit(nopython = True)
def monte_carlo_step(x):
    succrate = 0
    for _ in range(N):
        k = np.random.randint(N)
        kp, km = (k+1)%N, (k-1)%N
        x_probe = x[k] + 2*(np.random.random()-0.5)*de
        pi_a = rho_free(x[km], x[k])*rho_free(x[k], x[kp])*np.exp(-1/2*Dt*x[k]**2)
        pi_b = rho_free(x[km], x_probe)*rho_free(x_probe, x[kp])*np.exp(-1/2*Dt*x_probe**2)
        if np.random.random() < pi_a/pi_b:
            succrate += 1
            x[k] = x_probe
    succrate /= N

@nb.jit(nopython = True)
def monte_carlo(x, steps):
    out = []
    for _ in range(steps):
        newx = monte_carlo_step(x)
        out.append(np.average(out))
    return out
# %%
monte_carlo_step(x)
# %%
