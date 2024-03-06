'game theory, prisoners dillema 1'
#%% 
import numpy as np
import matplotlib.pyplot as plt
#%%
n = 201
'c -> 1'
'd -> 0'
space = np.ones((n,n))
space[int(n/2), int(n/2)] = 0
# %%
b = 1.9 
def switch(space, mval):
    directions = [(0,1), (0, -1), (1, 0), (-1,0 ), (1,1), (1,-1), (-1,1),(-1, 1), (-1,-1)]
    temp_diff = np.zeros(mval.shape)
    temp_type = np.zeros(mval.shape)-10e9
    for d in directions:
        diff = np.roll(mval, d, axis = (0,1))-mval
        idx = np.where(diff > temp_diff)
        temp_diff[idx] = diff[idx]
        temp_type[idx] = np.roll(space, d, axis = (0,1))[idx]
    idx2 = np.where(temp_type == -10e9)
    temp_type[idx2] = space[idx2]
    newspace = temp_type
    return newspace
#%%
def step(space, b):
    aggregation = np.zeros(space.shape)
    aggregation += np.roll(space, (0,1), axis = (0,1))
    aggregation += np.roll(space, (0,-1), axis = (0,1))
    aggregation += np.roll(space, (1,0), axis = (0,1))
    aggregation += np.roll(space, (-1,0), axis = (0,1))
    aggregation += np.roll(space, (1,1), axis = (0,1))
    aggregation += np.roll(space, (1,-1), axis = (0,1))
    aggregation += np.roll(space, (-1,1), axis = (0,1))
    aggregation += np.roll(space, (-1,-1), axis = (0,1))
    'pov c'
    m1 = (space == 1)*1*aggregation
    a1 = np.where(m1 > 0)
    m1[a1] += 1
    m2 = (space == 0)*b*aggregation
    valmap = m1+m2

    return switch(space, valmap)
# %%
nstep = 50
m = space
for s in range(nstep):
    m = step(m, b)
    plt.imshow(m[:,:], cmap='Greys', interpolation='nearest')
    plt.colorbar()
    plt.savefig(f'{s:06d}.png')
    plt.close()
# %%

plt.imshow(m[:,:], cmap='Greys', interpolation='nearest')
#%%   
temp = step(space, b)
# %%
temp
# %%
