from sklearn.datasets import load_boston

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

import numpy as np


# Load Boston housing data
print("Load Boston housing data ...")
boston = load_boston()



X = boston.data
y = boston.target


seed = 2018
np.random.seed(seed)




# First kernel: RBF
estimators_1 = []
estimators_1.append(('gp', GaussianProcessRegressor(kernel=RBF(), n_restarts_optimizer=9,normalize_y=True)))




pipeline_1 = Pipeline(estimators_1)


# In[10]:

print("Gaussian process using RBF kernel ...")
print("10-folder CV ...")

# 10-folder CV
kfold_1 = KFold(n_splits=10, random_state=seed)
results_1 = cross_val_score(pipeline_1, X, y, cv=kfold_1)


print("MSE using RBF kernel: %.2f " % (-1*results_1.mean()))




# Second kernel: Matern
estimators_2 = []
estimators_2.append(('gp', GaussianProcessRegressor(kernel=Matern(), n_restarts_optimizer=9,normalize_y=True)))
pipeline_2 = Pipeline(estimators_2)

print("Gaussian process using Matern kernel ...")
print("10-folder CV ...")

# 10-folder CV
kfold_2 = KFold(n_splits=10, random_state=seed)
results_2 = cross_val_score(pipeline_2, X, y, cv=kfold_2)
print("MSE using Matern kernel: %.2f " % (-1*results_2.mean()))









#~ # In[ ]:


