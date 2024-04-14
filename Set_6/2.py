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
h1 = np.random.randn(sample, N, N)
h2 = np.random.randn(sample, N, N)
h12 = (h1 + 1j*h2)/np.sqrt(2)
h_gue = 1/2*(h12+np.conjugate(np.transpose(h12, axes=(0,2,1))))
eig_gue = np.linalg.eigvals(h_gue)
eig_gue = eig_gue.flatten()
#%%
'goe'
h_goe = np.random.randn(sample, N, N)
h_goe = 1/2*(h_goe+np.transpose(h_goe, axes=(0,2,1)))
eig_goe = np.linalg.eigvals(h_goe)
eig_goe = eig_goe.flatten()
# %%
'slicer'
eig = eig_gue
# eig = eig_goe
n, data_bin, _ =  plt.hist(eig, 50, density=True, facecolor='cyan', alpha=0.75)
# center_bins = data_bin[len(data_bin)//2:len(data_bin)//2+2]
# center_data = eig[(eig >= center_bins[0]) & (eig <= center_bins[1])]
center_start = len(data_bin) // 4
center_end = 3 * (len(data_bin) // 4)
center_bins = data_bin[center_start:center_end]
center_data = eig[(eig >= center_bins[0]) & (eig <= center_bins[-1])]
seig = np.sort(center_data)
# idx = int(seig.shape[0]*3/4)
# sl = seig[idx:]
df = np.diff(seig)
df = df/np.average(df)
assert(np.allclose(np.imag(df), 0))
df = np.real(df)
#%%
'ploter'
n, bins, _= plt.hist(df, 50, density=True, facecolor='cyan', alpha=0.75, range =(0, 5))
goe = np.pi/2*bins*np.exp(-np.pi/4*bins**2)
gue = 32/np.pi**2*bins**2*np.exp(-4/np.pi*bins**2)
plt.plot(bins, gue, 'r-', linewidth=2)
# plt.plot(bins, goe, 'r-', linewidth=2)
plt.show()
# %%
