"""1 dokładność numeryczna"""
#%%
'epsilon maszynowy'
import numpy as np

# %%
values = [10**(-x) for x in range(30)]
a = 1
for x in values:
    print(a+x, "value:", np.log10(x))
    
"epsilon value dla 15"