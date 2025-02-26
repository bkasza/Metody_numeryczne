"""1 dokładność numeryczna"""
#%%
'epsilon maszynowy'
import numpy as np
import pandas as pd
# %%
values = [10**(-x) for x in range(30)]
a = 1
for x in values:
    print(a+x, "value:", int(np.log10(x)))
    
"epsilon value dla 15"
#%%
"kolejne wartosci ciagu"
x0 = 1
x1 = 1/3
n = 20
x_prev, x_cur = x0, x1
out = []
for i in range(n):
   x_new = 13/3*x_cur - 4/3*x_prev
   x_prev = x_cur
   x_cur = x_new
   corr = 1./3.**(i+2)
   out.append(np.array([x_new, corr, abs(corr-x_new)]))
   print("loop", x_new, "correct", corr, "diff", abs(corr-x_new))
#%%
'make df'
table = np.array(out)f}'.format)
dataset = pd.DataFrame({'loop': table[:, 0], 'cprrect': table[:, 1], 'diff': table[:, 2]})
dataset
# %%
