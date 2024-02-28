#%%%
'ants'
import numpy as np

# %%
# %%
def pbc(nx, ny, x, y):
    if x < nx:
        pass
    else:
        x = 0
    if y < ny:
        pass
    else:
        y = 0
    return [x, y]
# %%
class ant():
    def __init__(self, nx, ny, headpos = None, legpos = None):
        self.nx = nx
        self.ny = ny
        if headpos is None and legpos is None:
            self.headpos = np.array([int(nx/2), int(ny/2)])
            self.legpos = np.array([int(nx/2)+1, int(ny/2)])
        self.update_direction()
    def pbc(self):
        if self.headpos[0] >= self.nx:
            self.headpos[0] = 0
        if self.headpos[1] >= self.nx:
            self.headpos[1] = 0
        if self.legpos[0] >= self.ny:
            self.legpos[0] = 0
        if self.legpos[1] >= self.ny:
            self.legpos[1] = 0
    def update_direction(self):
        self.direction = self.headpos - self.legpos
    def move_white(self):
        'rotate head 90 to the right'
        if self.headpos[1] == self.legpos[1] and self.direction[0] < 0: 
            self.legpos = self.headpos 
            self.headpos = self.headpos + np.array([0, 1])
        elif self.headpos[1] == self.legpos[1] and self.direction[0] > 0: 
            self.legpos = self.headpos 
            self.headpos = self.headpos - np.array([0, 1])
        elif self.headpos[0] == self.legpos[0] and self.direction[1] < 0: 
            self.legpos = self.headpos 
            self.headpos = self.headpos - np.array([1, 0])
        elif self.headpos[0] == self.legpos[0] and self.direction[1] > 0: 
            self.legpos = self.headpos 
            self.headpos = self.headpos + np.array([1, 0])
        else:
            print('wrong')
        self.pbc()
    def move_black(self):
        'rotate head 90 to the left'
        if self.headpos[1] == self.legpos[1] and self.direction[0] < 0: 
            self.legpos = self.headpos 
            self.headpos = self.headpos - np.array([0, 1])
        elif self.headpos[1] == self.legpos[1] and self.direction[0] > 0: 
            self.legpos = self.headpos 
            self.headpos = self.headpos + np.array([0, 1])
        elif self.headpos[0] == self.legpos[0] and self.direction[1] < 0: 
            self.legpos = self.headpos 
            self.headpos = self.headpos + np.array([1, 0])
        elif self.headpos[0] == self.legpos[0] and self.direction[1] > 0: 
            self.legpos = self.headpos 
            self.headpos = self.headpos - np.array([1, 0])
        else:
            print('wrong')
        self.pbc()

# %%
nx, ny = 100, 100
antoni = ant(nx, ny)
space = np.zeros((nx, ny))
for i in range(10000):
    if space[antoni.headpos[0],antoni.headpos[1]] == 0:
        space[antoni.headpos[0],antoni.headpos[1]] = 1
        antoni.move_white()
    else:
        space[antoni.headpos[0],antoni.headpos[1]] = 0
        antoni.move_black()
    antoni.update_direction()
# %
# %%

'plot'
import matplotlib.pyplot as plt
plt.imshow(space[:,:])
# %%
'antkow 4'
nx, ny = 1000, 1000

ants = [ant(nx, ny), ant(nx, ny), ant(nx, ny), ant(nx, ny)]

space = np.zeros((nx, ny))
for i in range(100000):
    for a in ants:
        if space[a.headpos[0], a.headpos[1]] == 0:
            space[a.headpos[0], a.headpos[1]] = 1
            a.move_white()
        else:
            space[a.headpos[0], a.headpos[1]] = 0
            a.move_black()
        a.update_direction()
# %
# %%

'plot'
import matplotlib.pyplot as plt
plt.imshow(space[:,:], cmap='inferno')
plt.savefig('plocik.png', dpi = 700)
# %%
