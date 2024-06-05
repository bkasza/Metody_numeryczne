#%%
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import cm
# %%
'parameter definition'
'edge list'
D = 5
be = 10
dt = 0.01
edges = np.genfromtxt(r'zachary1.txt', dtype = int)
edges = np.concatenate((edges, edges[:, ::-1]), axis = 0)
weights = np.genfromtxt(r'zachary2.txt')/7
#%%
'function definitions'
def f(x):
    return (x-0.25)**3

def get_neigh_idx(idx, edges):
    neigh = []
    for e in edges:
        if e[0] == idx:
            neigh.append(e)
    return np.array(neigh)

def update_c(idx, edges, g):
    neigh = get_neigh_idx(idx, edges)
    vals = []
    c = g.nodes[idx]['state']
    for ne in neigh:
        vals.append(-(c - g.nodes[ne[1]]['state'])*g.edges[ne]['weight'])
    vals = np.array(vals)
    out = D * np.sum(vals)*dt
    return out

def update_w(w_idx, g):
    w = g.edges[w_idx]['weight']
    c_i = g.nodes[w_idx[0]]['state']
    c_j = g.nodes[w_idx[1]]['state']
    return -be*w*(1-w)*f(np.abs(c_i - c_j))*dt

def iter(g, nodes, edges):
    g_copy = g.copy()
    for n in nodes:
        if n == 0 or n == nodes[-1]:
            pass
        else:
            g.nodes[n]['state'] += update_c(n, edges, g_copy)
            g.nodes[n]['state'] = np.clip(g.nodes[n]['state'], 0, 1)
    for e in edges:
        g.edges[*e]['weight'] += update_w(e, g_copy)
        g.edges[*e]['weight'] = np.clip(g.edges[*e]['weight'], 0, 1)
    return g

def get_division(g, edges):
    N = edges.shape[0]
    val = 0
    for e in edges[:N//2+1, :]:
        c_i = g.nodes[e[0]]['state']
        c_j = g.nodes[e[1]]['state']
        if (c_i < 0.5 and c_j > 0.5) or (c_i > 0.5 and c_j < 0.5):
            val += g.edges[*e]['weight']
    return val

def plot(g, i):
    nx.draw_spring(g, cmap = cm.cool, vmin = 0, vmax = 1, with_labels = True, node_color =
    list(nx.get_node_attributes(g, "state").values()), edge_cmap = cm.binary, edge_vmin = 0,
    edge_vmax = 1, edge_color = list(nx.get_edge_attributes(g, "weight").values()))
    plt.savefig(f'{i:06d}.png')
    plt.close()

def loop(g, nodes, edges, steps, interval = 99):
    vals = []
    for s in range(steps):
        if s%interval == 0:
            plot(g, s)
        g = iter(g, nodes, edges)
        vals.append(get_division(g, edges))
    return vals
#%%
nodes = np.arange(34)
g = nx.Graph()
g.add_nodes_from(nodes)
g.add_edges_from(edges)
for n in nodes:
    if n == 0:
        g.nodes[n]['state'] = 0
    elif n == nodes[-1]:
        g.nodes[n]['state'] = 1    
    else:
        g.nodes[n]['state'] = 0.5
for e in edges:
    g.edges[*e]['weight'] = weights[*e]

# %%
'do stuff'
steps = 3000
vals = loop(g, nodes, edges, steps)
plt.plot(range(steps), vals)
# %%
