'antoni with border'
#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
#%%
class antoni():
    def __init__(self, pos, nx, ny):
        self.pos = pos
        self.food = False
        self.doordash = False
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
        val[0] = val[0]%self.nx
        val[1] = val[1]%self.ny
        return val
    def pbc(self):
        self.pos[0] = self.pos[0]%self.nx
        self.pos[1] = self.pos[1]%self.ny
    def far_away(self, no_longer = False):
        if no_longer:
            self.away = 0
        self.away += 1
    def find_next_dir(self):
        d = self.dir
        if d[0] == 0:
            return np.array([[1, d[1]],[0, d[1]],[-1, d[1]]])
        elif d[1] == 0:
            return np.array([[d[0], 1],[d[0], 0],[d[0], -1]])
        else: 
            return np.array([[0, d[1]], [d[0], d[1]], [d[0], 0]])
    def find_next_cell(self, space, home_fero, food_fero):
        self.doordash = False
        al = 5
        h = 2
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
                if any(space[ncells[:,0], ncells[:, 1]] == -1):
                    wall_idx = np.where(space[ncells[:,0], ncells[:, 1]] == -1)
                    wall_nidx = ncells[wall_idx]
                    mask = np.logical_not(np.isin(ncells, wall_nidx).all(axis=1))
                    result = ncells[mask]
                    ncells = result
                else:
                    pass
                if len(ncells) == 0:
                    ndir = -self.dir
                    new_cell = self.pos-self.dir
                else:        
                    hferons = np.copy(home_fero[ncells[:,0], ncells[:,1]])
                    hferons = (hferons + h)**al
                    hfero_tot = np.sum(hferons)
                    hferons = np.ones(len(ncells)) if hfero_tot == 0 else hferons
                    hfero_tot = 3 if hfero_tot == 0 else hfero_tot
                    prob = hferons/hfero_tot
                    new_idx = np.random.choice(np.arange(len(ncells)), 1, p = prob)  
                    new_cell = ncells[new_idx][0]
                    ndir = ndir[new_idx][0]
                    self.far_away()
        else:
            if any(space[ncells[:,0], ncells[:, 1]] == 2):
                new_idx = np.where(space[ncells[:,0], ncells[:, 1]] == 2)
                new_cell = ncells[new_idx][0]
                ndir = self.dir*(-1)
                self.food = False
                self.doordash = True
                self.far_away(no_longer=True)
            else: 
                if any(space[ncells[:,0], ncells[:, 1]] == -1):
                    wall_idx = np.where(space[ncells[:,0], ncells[:, 1]] == -1)
                    wall_nidx = ncells[wall_idx]
                    mask = np.logical_not(np.isin(ncells, wall_nidx).all(axis=1))
                    result = ncells[mask]
                    ncells = result
                else:
                    pass
                if len(ncells) == 0:
                    ndir = -self.dir
                    new_cell = self.pos
                else:
                    fferons = food_fero[ncells[:,0], ncells[:,1]]
                    fferons = (fferons + h)**al
                    ffero_tot = np.sum(fferons)
                    fferons = np.ones(len(ncells)) if ffero_tot == 0 else fferons
                    ffero_tot = 3 if ffero_tot == 0 else ffero_tot
                    prob = fferons/ffero_tot
                    new_idx = np.random.choice(np.arange(len(ncells)), 1, p = prob)  
                    new_cell = ncells[new_idx][0]
                    ndir = ndir[new_idx][0]
                    self.far_away()
        if self.food and len(ncells) > 0:
            home_fero[self.pos[0], self.pos[1]] += 0.99**self.away
        elif len(ncells) > 0:
            food_fero[self.pos[0], self.pos[1]] += 0.99**self.away
        else: 
            pass
        # print(self.pos)
        self.pos = new_cell
        self.pbc()
        self.dir = ndir
        return (space, home_fero, food_fero, self.doordash)
# %%
'big food chunk'
nx, ny = 80, 80
nx0, ny0 = 50, 60
space = np.zeros((nx, ny))
'add wall'
bar1 = [50, 70]
bar2 = [45, 45]
bar3 = [40, 60]
bar4 = [55, 55]
bar5 = [40, 40]
bar6 = [50, 60]
bar7 = [70, 70]
bar8 = [50, 60]
space[45, 50:70] = -1
# space[50:60, 40] = -1
# space[50:60, 70] = -1
space[55, 40:60] = -1
'def fero'
home_fero = np.zeros((nx, ny))
food_fero = np.zeros((nx, ny))
nfoodx = 40
nfoody = 35
'jedzenie'
# space[:nfoodx, :nfoody] = 1
space[30:70, 0:35] = 1
'chata'
space[nx0, ny0] = 2

#%%
mrowisko = []
antosie = 80
def gen_ants():
    for i in range(antosie):
        mrowisko.append(antoni([nx0,ny0], 80,80))
        # print(mrowisko[i].dir)
gen_ants()# %%
step = 3000
foodcount = 0
totfood = nfoodx*nfoody
for i in range(step):
    posx = []
    posy = []
    for antek in mrowisko:
        a = antek.find_next_cell(space, home_fero, food_fero)
        space = a[0]
        home_fero = a[1]
        food_fero = a[2]
        if a[3]:
            foodcount += 1
        posx.append(antek.pos[0])
        posy.append(antek.pos[1])
        # print(antek.pos)
    home_fero = home_fero * 0.99
    food_fero = food_fero * 0.99
    if i%50 == 0:
        plt.imshow(home_fero+food_fero)
        plt.scatter(posy,posx, marker='.', color = 'black')
        plt.plot(bar1, bar2, 'b-')
        plt.plot(bar3, bar4, 'b-')
        # plt.plot(bar5, bar6, 'b-')
        # plt.plot(bar7, bar8, 'b-')
        plt.gcf().set_facecolor("pink")
        plt.title(f'step {i}, food: {foodcount}, totfodd: {totfood}')
        plt.savefig(f'{i:06d}.png')
        plt.close()
    # %%
# plt.imshow(np.log(home_fero))
# plt.scatter(posy,posx, marker='.', color = 'red', vmin=0, vmax = 80)
plt.imshow(space)
plt.savefig('plot.png', dpi = 300)
plt.colorbar()
plt.show()
# %%
