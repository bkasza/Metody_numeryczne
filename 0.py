'test commit, jupyter lense'

#%%
import numpy as np
from scipy.linalg import expm
a = np.random.random((100, 100))
S = expm(1j*a)

# %%
import matplotlib.pyplot as plt

# %%
plt.imshow(np.real(S))
# %%
