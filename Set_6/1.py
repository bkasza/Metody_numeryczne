#%%
'random matrix theorem'
import numpy as np
import matplotlib.pyplot as plt
# %%
'goe ensamble'
bet = 1
N = 8
sample = 2000
R = np.sqrt(2*N)
h = np.random.randn(sample, N, N)
h = 1/2*(h+np.transpose(h, axes=(0,2,1)))
eig = np.linalg.eigvals(h).flatten()

# %%
'ploter'
n, bins, _= plt.hist(eig, 50, density=True, facecolor='cyan', alpha=0.75)
wigner = 2/(np.pi*R**2)*np.sqrt(R**2-bins**2)
plt.plot(bins, wigner, 'r-', linewidth=2)
plt.show()
# %%
