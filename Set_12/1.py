"neural network"
# %%
import numpy as np
import matplotlib.pyplot as plt
import random

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
def check_pattern(W, x, steps):
    x_0 = x.copy()
    for i in range(steps):
        Wx = W @ x
        x = np.sign(Wx)
    return "Patern the same " + f"{np.allclose(x, x_0)}"


"flip random 5 and convolve"
def random_flip(x, n=5):
    idx = np.array(random.sample(range(nx * ny), n))
    x[idx] *= -1
    return x


def ploter(x):
    for x in x:
        plt.imshow(x.reshape(nx, ny))
        plt.show()
        plt.clf()


def recognition(W, x, steps):
    x_0 = x.copy()
    for i in range(steps):
        Wx = W @ x
        x = np.sign(Wx)
    return x

def plot_together(bottom_data, top_data):
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    axs = axs.flatten()
    for i in range(8):
        if i < 4:
            data = bottom_data[i%4].reshape(7,5)
            axs[i].imshow(data)
            axs[i].set_title(f'messed {i%4}')
        else:
            data = top_data[i%4].reshape(7,5)
            axs[i].imshow(data)
            axs[i].set_title(f'fixed {i%4}')
    plt.tight_layout()
    plt.show()

# %%
pat = np.array([P1, P2, P3, P4])
x = [x.flatten() * 2 - 1 for x in pat]
out = np.array([np.outer(x, x) for x in x])
W = np.average(out, axis=0) - np.eye(nx * ny, nx * ny)
messed = [random_flip(x, n=20) for x in x]
fixed = [recognition(W, m, 5) for m in messed]
plot_together(messed, fixed)
# %%
