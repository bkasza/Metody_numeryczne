"""1 dokładność numeryczna"""
#%%
'epsilon maszynowy'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %%
values = [10**(-x) for x in range(30)]
a = 1
for x in values:
    print(a+x, "value:", int(np.log10(x)))
    
"epsilon value dla 15"
#%%
"kolejne wartosci ciagu"
x0 = 1
x1 = 1./3.
n = 20
x_prev, x_cur = x0, x1
out = []
for i in range(n):
   x_new = 13./3.*x_cur - 4./3.*x_prev
   x_prev = x_cur
   x_cur = x_new
   corr = 1./3.**(i+2)
   out.append(np.array([x_new, corr, abs(corr-x_new)]))
   print("loop", x_new, "correct", corr, "diff", abs(corr-x_new))
#%%
'make df'
table = np.array(out)
dataset = pd.DataFrame({'loop': table[:, 0], 'correct': table[:, 1], 'diff': table[:, 2]})
dataset
# %%
"niech ta funkcja se zbiegnie"
def a_n(n_max):
    out = []
    for n in range(1, n_max):
        out.append((n+1)/n)
    return out
def b_n(n_max):
    out = []
    for n in range(1, n_max):
        out.append((1+1/n)**n)
    return out
def c_n(n_max, c0 = 2):
    out = []
    for n in range(0, n_max-1):
        if n == 0:
            curr = c0
        else:
            curr = 1/2*curr+1/curr
        out.append(curr)
    return out
def d_n(n_max, d0 = 0, d1 = 2):
    out = [d0, d1]
    d_prev = d0
    d_curr = d1
    for n in range(2, n_max-1):
        curr = (d_prev*d_curr+1)/(d_prev + d_curr)
        d_prev = d_curr
        d_curr = curr
        out.append(curr)
    return out
n_max = 1000
n = np.arange(1,n_max)
fig, axes = plt.subplots(4,1, figsize=(6, 16))
axes[0].plot(n, a_n(n_max), label = "a_n")
axes[0].axhline(1, color = 'r')
axes[1].plot(n, b_n(n_max), label = "b_n")
axes[1].axhline(np.e, color = 'r')
axes[2].plot(n, c_n(n_max), label = "c_n")
axes[2].axhline(np.sqrt(2), color = 'r')
axes[3].plot(n, d_n(n_max), label = "d_n")
axes[3].axhline(1, color = 'r')
for ax in axes:
    ax.legend()
    ax.set_ylabel('approximated values')
    ax.set_xlabel('n')
    ax.set_yscale("log")
    ax.set_xscale("log")
# %%
