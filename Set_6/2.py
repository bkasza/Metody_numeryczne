#%%
'random matrix theorem'
import numpy as np
import matplotlib.pyplot as plt
# %%
'task 2'
N = 200
sample = 500
R = np.sqrt(2*N)
'gue'
def get_gue():
    h1 = np.random.randn(sample, N, N)
    h2 = np.random.randn(sample, N, N)
    h12 = (h1 + 1j*h2)/np.sqrt(2)
    h_gue = 1/2*(h12+np.conjugate(np.transpose(h12, axes=(0,2,1))))
    eig_gue = np.linalg.eigvals(h_gue)
    return eig_gue

'goe'
def get_goe():
    h_goe = np.random.randn(sample, N, N)
    h_goe = 1/2*(h_goe+np.transpose(h_goe, axes=(0,2,1)))
    eig_goe = np.linalg.eigvals(h_goe)
    return eig_goe
# %%
'slicer'
# eig = get_gue()
eig = get_goe()
assert(np.allclose(np.imag(eig), 0))
eig = np.real(eig)
#%%
def take_quarter(arr):
    quarter_length = len(arr) // 4
    start_index = len(arr) // 2 - quarter_length // 2
    end_index = start_index + quarter_length
    return arr[start_index:end_index]

def take_two_middle_values(arr):
    n = len(arr)
    if n % 2 == 0:
        return arr[n // 2 - 1 : n // 2 + 1]
    else:
        return [arr[n // 2]]

def slicer(arg):
    sarg = np.sort(arg)
    if N >= 10:
        return np.diff(take_quarter(sarg))
    else:
        return np.diff(take_two_middle_values(sarg))
df = []
for e in eig:
    df.append(slicer(e.flatten()))
df = df/np.average(df)
#%%
'ploter'
n, bins, _= plt.hist(df, 50, density=True, facecolor='cyan', alpha=0.9, range =(0, 5))
goe = np.pi/2*bins*np.exp(-np.pi/4*bins**2)
gue = 32/np.pi**2*bins**2*np.exp(-4/np.pi*bins**2)
# plt.plot(bins, gue, 'r-', linewidth=2)
plt.plot(bins, goe, 'r-', linewidth=2)
plt.show()
# %%
