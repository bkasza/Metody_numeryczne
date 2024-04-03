# %%
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_binary_string(n=512):
    number = random.getrandbits(n)
    binary_string = format(number, "0{}b".format(n)).zfill(512)
    return np.array(list(binary_string))
checkerboard = np.zeros((50, 50), dtype=int)
checkerboard[::2, ::2] = 1  
checkerboard[1::2, 1::2] = 1 

# %%
nx, ny = 50, 50
n_chromos = 10
n_space = 5
iter_steps = 100
evol_step = 100
chromos = np.array([generate_binary_string() for i in range(n_chromos)])
direction = [
    (-1, 1),
    (0, 1),
    (1, 1),
    (-1, 0),
    (0, 0),
    (1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
]


# %%
def bit_flip(x):
    if x == '0': 
        return '1'
    if x == '1':
        return '0'
    raise 'wtf'
def calc_N(grid):
    N = np.zeros(grid.shape, dtype=int)
    for p, d in enumerate(direction):
        N += 2**p * np.roll(grid, d, axis=(0, 1))
    return N


def rule(grid, gene):
    N = calc_N(grid)
    idx = N.flatten()
    # gene = np.fromstring(gene, np.int8) - 48
    # assert len(gene) == 512
    out = gene[idx.astype(int)]
    return out.reshape(nx, ny)


def iter_grid(space, chromos):
    for g in range(n_space):
        for i in range(iter_steps):
            space[g] = rule(space[g], chromos)
    return space


def change_chromos(chromos, fitness):
    # print(chromos.shape)
    med = np.median(fitness)
    idx = np.where(fitness > med)
    if med > 0:
        pass
    else: 
        fitness[np.where(fitness < 0)] = 0    
    weight = fitness[idx] / np.sum(fitness[idx])
    lived = chromos[idx]
    kids = np.array([])
    if len(idx[0] > 2):
        cl = chromos.shape[0] - len(idx[0]) - 2
        for k in range(2):
            crossoverpoint = np.random.randint(512)
            parents = random.sample(range(0, len(idx[0])), 2)
            kid = (np.concatenate((
                chromos[idx][parents[0]][:crossoverpoint],
                chromos[idx][parents[1]][crossoverpoint:]))
            )
            flip = np.random.choice(np.arange(512), size = np.random.randint(3))
            for f in flip:
                kid[f] = bit_flip(kid[f])
            kids = np.append(
                kids,
                kid,
            )
        kids.shape = 2, 512
    else:
        cl = chromos.shape[0] - len(idx[0])
    cloned = chromos[np.random.choice(idx[0], size=cl, p=weight)]
    out = np.concatenate((lived, kids, cloned))
    return out

def calc_fitness(grid):
    f = 0
    comp1 = np.roll(grid, (1, 0), (0, 1))
    count = np.sum((comp1 == grid) * 1)
    comp2 = np.roll(grid, (0, 1), (0, 1))
    count += np.sum((comp2 == grid) * 1)
    f -= 3 * count
    comp3 = np.roll(grid, (1, 1), (0, 1))
    count = np.sum((comp3 == grid) * 1)
    f += count * 8 - (nx * ny - count) * 5
    comp4 = np.roll(grid, (1, -1), (0, 1))
    count = np.sum((comp4 == grid) * 1)
    f += count * 8 - (nx * ny - count) * 5
    return f

def fitness(space, chromos):
    f_seq = []
    space = np.array([np.random.randint(0, 2, (nx, ny)) for i in range(n_space)])
    for ch in chromos:
        spacex = iter_grid(space, ch)
        for s in space:
            f = calc_fitness(s)
        f_seq.append(f / spacex.shape[0])
    return np.array(f_seq)


def evol(space, chromos):
    f_seq = fitness(space, chromos)
    # print(f_seq)
    chromos = change_chromos(chromos, f_seq)
    return f_seq, chromos

# %%

out_f = []

for ev in tqdm(range(evol_step)):
    f_seq, chromos = evol(space, chromos)
    out_f.append(np.max(f_seq)/np.sum(f_seq))

#%%
plt.plot(np.arange(evol_step), out_f)
# %%
