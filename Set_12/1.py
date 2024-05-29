"neural network"
# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
nx, ny = 7, 5
P1 = [
    [1, 1, 1, 1, 0],
    [0, 1, 0, 0, 1],
    [0, 1, 0, 0, 1],
    [0, 1, 0, 0, 1],
    [0, 1, 0, 0, 1],
    [0, 1, 0, 0, 1],
    [0, 1, 1, 1, 0],
]
P2 = [
    [1, 1, 1, 1, 1],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 0, 0, 1, 0],
    [1, 1, 1, 0, 0],
]
P3 = [
    [0, 1, 1, 1, 1],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 1, 1, 1, 1],
]
P4 = [
    [1, 0, 0, 0, 1],
    [1, 1, 0, 1, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
]
P1 = np.array(P1)
P2 = np.array(P2)
P3 = np.array(P3)
P4 = np.array(P4)
# %%
pat = np.array([P1, P2, P3, P4])
x = [x.flatten()*2-1 for x in pat]
out = np.array([np.outer(x, x) for x in x])
# %%
W = np.average(out, axis = 0)-np.eye(nx*ny, nx*ny)
# %%
plt.imshow(W)
plt.colorbar()
# %%
def check_pattern(W, x, steps):
    for i in range(steps):
        x_new = W @ x
        x = x_new
    return 'Patern the same '+f'{np.allclose(x_new, x)}'
check_pattern(W, x[3], 5)
# %%
