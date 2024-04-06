# %%
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm


def generate_binary_string(n=512):
    return np.random.randint(2, size=n)


def generate_space(nx, ny):
    space = np.array([np.random.randint(0, 2, (nx, ny)) for i in range(n_space)])
    return space


chb = np.zeros((50, 50), dtype=int)
chb[::2, ::2] = 1
chb[1::2, 1::2] = 1
chb = chb[None, :]
# %%
n_binary = 512
nx, ny = 50, 50
n_chromos = 20
iter_steps = 100
evol_step = 300
n_space = 10
chromos = np.array([generate_binary_string(n=n_binary) for i in range(n_chromos)])
direction = [
    (1, 1),
    (1, 0),
    (1, -1),
    (0, 1),
    (0, 0),
    (0, -1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
]

#%%
def bit_flip(x):
    if x == 0:
        return 1
    if x == 1:
        return 0
    raise "wtf"


def calc_N(grid):
    N = np.zeros(grid.shape, dtype=int)
    for p, d in enumerate(direction):
        N += 2**p * np.roll(grid, d, axis=(0, 1))
    return N


def rule(grid, gene):
    N = calc_N(grid)
    idx = N.flatten()
    out = gene[idx.astype(int)]
    return out.reshape(nx, ny)


def iter_grid(space, chromos):
    for g in range(n_space):
        for i in range(iter_steps):
            space[g] = rule(space[g], chromos)
    return space

def iter_grid_plot(grid, chrom, t_step = 5):
    for i in range(iter_steps):
        grid = np.int32(rule(grid, chrom))
        if i%t_step == 0:
            plt.imshow(grid)
            plt.title(f'krok: {i}')    
            plt.savefig(f'{i:06d}.png')
            plt.close()

def change_chromos(chromos, fitness):
    if np.any(fitness < 0):
        fit_min = np.min(fitness)
        fitness = fitness + np.abs(fit_min) + 1
    med = np.median(fitness)
    idx = np.where(fitness >= med)
    weight = fitness[idx] / np.sum(fitness[idx])
    lived = chromos[idx][:5]
    cloned = chromos[np.random.choice(idx[0], size=5, p=weight)]
    kids = np.array([])
    n_kids = 10
    for k in range(n_kids):
        crossoverpoint = np.random.randint(n_binary)
        parents = random.sample(range(0, 5), 2)
        kid = np.concatenate(
            (
                chromos[idx][parents[0]][:crossoverpoint],
                chromos[idx][parents[1]][crossoverpoint:],
            )
        )
        flip = np.random.choice(np.arange(n_binary), size=np.random.randint(3))
        for f in flip:
            kid[f] = bit_flip(kid[f])
        kids = np.append(
            kids,
            kid,
        )
    kids.shape = n_kids, n_binary
    out = np.concatenate((lived, kids, cloned))
    assert out.shape[0] == n_chromos
    return out


def calc_fitness(space):
    f = 0
    for grid in space:
        comp1 = np.roll(grid, (0, -1), (0, 1)) == grid
        count = np.sum(comp1 * 1)
        comp2 = np.roll(grid, (1, 0), (0, 1)) == grid
        count += np.sum(comp2 * 1)
        f -= 3 * count
        cond = comp1 + comp2
        idx = np.where(cond == False)
        comp3 = np.roll(grid, (1, -1), (0, 1))
        count = np.sum((comp3 == grid)[idx] * 1)
        f += count * 8 - (len(idx[0]) - count) * 5
        comp4 = np.roll(grid, (-1, -1), (0, 1))
        count = np.sum((comp4 == grid)[idx] * 1)
        f += count * 8 - (len(idx[0]) - count) * 5
    return f / space.shape[0]


def fitness(space, chromos):
    f_seq = []
    for ch in chromos:
        space_evol = iter_grid(space, ch)
        f = calc_fitness(space_evol)
        f_seq.append(f)
    # print(f_seq)
    return np.array(f_seq)


def evol(space, chromos):
    f_seq = fitness(space, chromos)
    chromos = change_chromos(chromos, f_seq)
    return f_seq, chromos


# %%
'ewolucja genetyczna'
out_f = []
for ev in tqdm(range(evol_step)):
    f_seq, chromos = evol(generate_space(nx, ny), chromos)
    out_f.append(np.average(f_seq))

# %%
plt.plot(np.arange(len(out_f)), out_f)
# %%
'foto ewolucja grida dla zadanego genomu'
m = generate_space(nx, ny)
iter_grid_plot(m[0], chromos[0])
# %%
