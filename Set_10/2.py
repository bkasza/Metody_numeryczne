#%%
'monte carlo with numba, full'
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
#%%
N = 40
be = 10
de = 0.615
Dt = be/N
print('de', de)
# %%

def pi(x, be):
    sigma2 = 1/(2*np.tanh(be/2))
    return 1/np.sqrt(2*np.pi*sigma2)*np.exp(-x**2/(2*sigma2))

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
        if np.random.random() < pi_b/pi_a:
            succrate += 1
            x[k] = x_probe
    succrate /= N
    return x

@nb.jit(nopython = True)
def monte_carlo(x, steps):
    out_avg = []
    out_var = []
    x0_seq = []
    xN2_seq = []
    for _ in range(steps):
        newx = monte_carlo_step(x)
        out_avg.append(np.average(newx))
        out_var.append(np.var(newx))
        x0_seq.append(newx[0])
        xN2_seq.append(newx[N//2])
    return out_avg, out_var, x0_seq, xN2_seq
# %%
x = np.linspace(-de, de, N)
nstep = 1000000
out_avg, out_var, x0_seq, xN2_seq = monte_carlo(x, nstep)
# x0_seq, xN2_seq = np.array(x0_seq), np.array(xN2_seq)
# %%
plt.plot(range(nstep), out_avg, label = 'avg')
plt.axhline(y=np.average(out_avg), color='r', linestyle='-', label = 'msc avg avg')
plt.legend()
plt.show()
plt.clf()
plt.plot(range(nstep), out_var, label = 'var')
plt.axhline(y=np.average(out_var), color = 'r',  label = 'mcs avg var')
plt.legend()
plt.show()
# %%
plt.hist(x0_seq+xN2_seq, density = True, bins = 100, label = 'MC')
x = np.linspace(-3, 3, N)
plt.plot(x, pi(x, be), label = 'theory')
plt.legend()
# %%
