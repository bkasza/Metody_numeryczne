#%%
import numpy as np
# %%
nx, ny = 100, 100
space = np.random.randint(0, 2, (nx, ny))
# space = np.zeros((nx, ny))
space[1:4, 2] = 1
space[2, 3] = 1
space[0:, 0] = 0
space[0:, ny-1] = 0
space[0, 0:] = 0
space[nx-1, 0:] = 0

# %%
generations = 1000
'zasady'
'3 sasiedzi moora -> nowa zywa'
'2 lub 3 dla zywej -> przezywa'
'pp -> umiera'
#%%
import matplotlib.pyplot as plt
plt.imshow(space[:,:], cmap='Greys')
plt.savefig('plocik.png', dpi = 300)
plt.close()
# %%
def keep_on_rollin_baby(space):
    aggregation = np.zeros(space.shape)
    aggregation += np.roll(space, (0,1), axis = (0,1))
    aggregation += np.roll(space, (0,-1), axis = (0,1))
    aggregation += np.roll(space, (1,0), axis = (0,1))
    aggregation += np.roll(space, (-1,0), axis = (0,1))
    aggregation += np.roll(space, (1,1), axis = (0,1))
    aggregation += np.roll(space, (1,-1), axis = (0,1))
    aggregation += np.roll(space, (-1,1), axis = (0,1))
    aggregation += np.roll(space, (-1,-1), axis = (0,1))
    idx1 = np.where(aggregation == 3)
    out2 = (space == 1)*(aggregation == 2)*1
    out1 = np.zeros(space.shape)
    out1[idx1] = 1
    out = out1+out2
    out[0:, 0] = 0
    out[0:, ny-1] = 0
    out[0, 0:] = 0
    out[nx-1, 0:] = 0
    return out
# %%
out = space
import matplotlib.pyplot as plt
for g in range(generations):
    out = keep_on_rollin_baby(out).reshape(nx,ny)
    plt.imshow(out[:,:], cmap='Greys', interpolation='nearest')
    plt.savefig(f'plocik{g}.png')
    plt.close()

# %%
plt.imshow(out[:,:], cmap='Greys', interpolation='nearest')
plt.savefig(f'plocik.png')
# %%
