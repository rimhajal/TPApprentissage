#%%
import numpy as np

n,p=9,6
A=np.random.randn(50,n,p)
U_fat, s, V_fat = np.linalg.svd(A,full_matrices=False)
# %%
Ainv = np.linalg.pinv(A)

# %%
