"neural network"
# %%
import numpy as np
import matplotlib.pyplot as plt
import random

# %%
def single_bit_flip(x):
    idx = np.random.randint(len(x))
    x[idx] *=-1
    return x
def check_pattern(W, x, steps):
    x_0 = x.copy()
    for i in range(steps):
        Wx = W @ x
        x = np.sign(Wx)
    return "Patern the same " + f"{np.allclose(x, x_0)}"

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
J = 100
def extra(N):
    x = np.random.randint(2, size = (N, J))*2-1
    out = np.array([np.outer(x, x) for x in x])
    W = np.average(out, axis=0) - np.eye(J,J)
    'do one bit flip for x'
    x_per = np.array([single_bit_flip(x_i) for x_i in x])
    x_stable = np.array([recognition(W, m, 1000) for m in x_per]) 
    avg = np.average(np.einsum('ai, ai -> a',x_per, x_stable)/J, axis = 0)
    return avg
N_seq = np.array(range(1, 50))
x = N_seq/J
y = [extra(n) for n in N_seq]
plt.plot(x, y)
# %%
