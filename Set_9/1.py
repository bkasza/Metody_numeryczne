'qba is back'
#%%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
#%%
'param definition'
L = 100
#%%
'function definition'


def neighbours(i):
    ix, iy = i
    return [ ( ix, (iy+1) % L ), ( ix, (iy-1) % L ),
            ( (ix+1) % L, iy ), ( (ix-1) % L, iy ) ]

def update(i):
    s[i] = 1
    for j in neighbours(i):
        h_loc[j] += 2

def find_unflip_positive(i, H):
    nidx = neighbours(i)
    out = []
    for idx in nidx:
        if s[idx] == -1 and h_loc[idx] + H > 0:
            out.append(idx)
    return out

def do_aval():
    i = 0
    while(i < 1):
        i_trig = np.unravel_index(np.argmax(h_loc + (s+1)*(-100)), h_loc.shape)
        H = - h_loc[i_trig]
        d = []
        d.append(i_trig)
        flipped = []
        while len(d) > 0:
            itmp = d.pop(0)
            if s[itmp] == -1: 
                update(itmp)
                d += find_unflip_positive(itmp, H)    
        flipped.append((np.sum(s)+L**2)/2)
        i += 1
    return flipped[0]
# %%
'task 1'
out = []
nsample = 1000
R = 1.4
for i in range(nsample):
    s = np.ones((L, L), dtype = int)*-1
    h_rnd = np.random.randn(L, L) * R
    h_loc = np.ones( (L, L) ) * (-4.0) + h_rnd
    out.append(do_aval())
out = np.array(out)
print(f'R: {R}, avg: {np.sum(out)/nsample}({(np.std(out)/np.sqrt(nsample)).round(3)})')
#%%

# %%
