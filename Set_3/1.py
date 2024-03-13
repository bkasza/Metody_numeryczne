'antonis'
#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
#%%
class antoni():
    def __init__(self, pos, nx, ny):
        self.pos = pos
        self.food = False
        self.nx = nx
        self.ny = ny
        self.init_dir()
        self.away = 0
    def init_dir(self):
        self.dir = [int(np.random.random()*3)-1, int(np.random.random()*3)-1]
        if self.dir == [0,0]:
            self.init_dir()
        assert(self.dir[0] != 0 or self.dir[1] != 0)
    def find_food(self):
        self.food = True
    def pbc_any(self, val):
        if np.abs(val[0]) == self.nx:
            val[0] = 0
        if np.abs(val[1]) == self.ny:
            val[1] = 0
        return [val[0], val[1]]
    def pbc(self):
        if self.pos[0] == self.nx:
            self.pos[0] = 0
        if self.pos[1] == self.ny:
            self.pos[1] = 0
    def far_away(self, no_longer = False):
        if no_longer:
            self.away = 0
        self.away += 1
    def find_next_dir(self):
        d = self.dir
        if d[0] == 0:
            return np.array([[1, d[1]],[0, d[1]],[-1, d[1]]])
        elif d[1] == 0:
            return np.array([[d[0], 1],[d[1], 0],[d[1], 1]])
        else: 
            return np.array([[0, d[1]], [d[0], d[1]], [d[0], 0]])
    def find_next_cell(self, space, home_fero, food_fero):
        al = 5
        h = 4
        ndir = self.find_next_dir()
        ncells = ndir + np.array(self.pos)
        for i in range(ncells.shape[0]):
            ncells[i] = self.pbc_any(ncells[i])
        if self.food == False:
            if any(space[ncells[:,0], ncells[:, 1]] == 1):
                new_idx = np.where(space[ncells[:,0], ncells[:, 1]] == 1)
                new_cell = ncells[new_idx][0]
                self.food = True
                space[new_cell[0], new_cell[1]] = 0
                ndir = self.dir*(-1)
                self.far_away(no_longer=True)
            else:
                hferons = np.copy(home_fero[ncells[:,0], ncells[:,1]])
                hferons = (hferons + h)**al
                hfero_tot = np.sum(hferons)
                hferons = np.ones(len(ncells)) if hfero_tot == 0 else hferons
                hfero_tot = 3 if hfero_tot == 0 else hfero_tot
                prob = hferons/hfero_tot
                new_idx = np.random.choice(np.arange(3), 1, p = prob)  
                new_cell = ncells[new_idx][0]
                ndir = ndir[new_idx][0]
        else:
            if any(space[ncells[:,0], ncells[:, 1]] == 2):
                new_idx = np.where(space[ncells[:,0], ncells[:, 1]] == 2)
                new_cell = ncells[new_idx][0]
                ndir = self.dir*(-1)
                self.food = False
            else: 
                fferons = food_fero[ncells[:,0], ncells[:,1]]
                fferons = (fferons + h)**al
                ffero_tot = np.sum(fferons)
                fferons = np.ones(len(ncells)) if ffero_tot == 0 else fferons
                ffero_tot = 3 if ffero_tot == 0 else ffero_tot
                prob = fferons/ffero_tot
                new_idx = np.random.choice(np.arange(3), 1, p = prob)  
                new_cell = ncells[new_idx][0]
                ndir = ndir[new_idx][0]
        if self.food:
            home_fero[self.pos[0], self.pos[1]] += 1
        else:
            food_fero[self.pos[0], self.pos[1]] += 1/(1+self.away*0.01)
        self.pos = new_cell
        self.pbc()
        return (space, home_fero, food_fero)
# %%
'big food chunk'
nx, ny = 80, 80
nx0, ny0 = 70, 70
space = np.zeros((nx, ny))
home_fero = np.zeros((nx, ny))
food_fero = np.zeros((nx, ny))
nfoodx = 40
nfoody = 40
'jedzenie'
space[:nfoodx, :nfoody] = 1
'chata'
space[nx0, ny0] = 2
mrowisko = []
antosie = 100
def gen_ants():
    for i in range(antosie):
        mrowisko.append(antoni([nx0,ny0], 80,80))
        # print(mrowisko[i].dir)
gen_ants()# %%
step = 1000
for i in range(step):
    for antek in mrowisko:
        a = antek.find_next_cell(space, home_fero, food_fero)
        space = a[0]
        home_fero = a[1]
        food_fero = a[2]
    home_fero = home_fero * 0.99
    food_fero = food_fero * 0.99
    # if i%10 == 0:
    #     plt.imshow(home_fero+food_fero)
    #     plt.savefig(f'{i:06d}.png')
# %%
# plt.imshow(np.log(home_fero))
plt.imshow(food_fero)
plt.savefig('plot.png', dpi = 300)
plt.colorbar()
# %%
