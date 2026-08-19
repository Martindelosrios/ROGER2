# # Needed libraries

# +
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn as sk
import pandas as pd
from scipy.stats import binned_statistic_2d
import seaborn as sns
import emcee
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from itertools import product
from tqdm import tqdm

from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from mlxtend.plotting import plot_confusion_matrix

from joblib import dump, load

from pyROGER import roger
from pyROGER import models
# -

models.list_saved_models()

import importlib
with importlib.resources.path("pyROGER", "dataset") as p:
    DATA_PATH = str(p)

# !ls '/home/mdelosrios/trabajos/pyROGER/pyROGER/dataset'

# ## Custom functions

# +
# Color scheme
cl_col  = 'red'
bs_col  = 'orange'
rin_col = 'green'
inf_col = 'blue'
itl_col = 'gray'

colors = [cl_col, bs_col, rin_col, inf_col, itl_col]
labels = ['CL', 'BS', 'RIN', 'IN', 'ITL']
# -

# # Reading data

DATA_PATH = '../data/hernan/'

# !ls ../data/hernan

# +
# data

data = np.loadtxt(DATA_PATH + 'galaxies_13_08_26.dat')

'''
Columna 1 y 2 nombres
columna 3: masa total
columna 4: r/r200
columna 5: delta V/sigma
'''
# -

print(np.min(data, axis = 0))
print(np.max(data, axis = 0))

data[:,2:5].shape

# !ls ../data/modelos_roger2

# +
path = '../data/modelos_roger2'
niter = 10

pred_prob_list = []
for i in range(niter):
  print(i)

  #roger2_trained = roger.RogerModel(x_dataset = data_train[random_ind, 2:], y_dataset = data_train[random_ind, 1],
   #                         ml_models = [KNeighborsClassifier(n_neighbors=63), RandomForestClassifier(max_depth=2, random_state=0)])
  roger2 = models.Roger2
  roger2.train(path_to_saved_model = [path + f'/roger2_KNN_{i}.joblib', path + f'/roger2_RF_{i}.joblib'])

  pred_prob_list.append( roger2.predict_prob(data[:,2:5], n_model = 0) )
# -

pred_prob_list = np.asarray(pred_prob_list)
pred_prob = np.mean(pred_prob_list, axis = 0)
pred_prob_sd = np.std(pred_prob_list, axis = 0)

# +
readme = '''
         Results corresponding to KNN method of ROGERv2.

         Columns:
         -------
         
         nombre1: 
         nombre2: 
         LogM: Log10 of the cluster mass.
         R/R200: Galaxy radial distance to the cluster center, normalized to R200.
         V/sigma: Galaxy relative velocity to cluster center normalized to cluster velocity dispersion.
         P_cl: Probability of being a cluster galaxy.
         P_bs: Probability of being a backsplash galaxy.
         P_rin: Probability of being a recent infaller galaxy.
         P_in: Probability of being an infalling galaxy.
         P_itl: Probability of being a iterloper galaxy.
         P_cl_std: Standard Deviation probability of being a cluster galaxy.
         P_bs_std: Standard Deviation probability of being a backsplash galaxy.
         P_rin_std: Standard Deviation probability of being a recent infaller galaxy.
         P_in_std: Standard Deviation probability of being an infalling galaxy.
         P_itl_std: Standard Deviation probability of being a iterloper galaxy.
         '''
np.savetxt('../data/hernan/ROGERv2_KNN_probabilities.txt',  np.hstack((data, pred_prob, pred_prob_sd)),
          header = 'ID1 ID2 LogM R/R200 V/sigma P_cl P_bs P_rin P_in P_itl P_cl_std P_bs_std P_rin_std P_in_std P_itl_std',
          comments = readme)

#pr = np.loadtxt('../data/hernan/ROGERv2_KNN_probabilities.txt', skiprows = 22)
# -

# # Analysis

max_class = np.argmax(pred_prob,axis=1)


plt.hist(max_class)

# +
fig,ax = plt.subplots(1,5, sharex = True, sharey = True, figsize = (12,3))

for i in range(5):
    ind = np.where(max_class == i)[0]
    ax[i].scatter(data[ind,3], data[ind,4], c = colors[i])
    ax[i].text(0.7,0.8, '{:.2f}%'.format(len(ind)/len(data)), transform = ax[i].transAxes)

ax[0].set_xlim(0,3)
ax[0].set_ylim(0,3)

ax[0].set_xlabel('$R/R_{200}$')
ax[1].set_xlabel('$R/R_{200}$')
ax[2].set_xlabel('$R/R_{200}$')
ax[3].set_xlabel('$R/R_{200}$')
ax[4].set_xlabel('$R/R_{200}$')

ax[0].set_ylabel('$\Delta V/ \sigma$')

plt.savefig('../graphs/porcentajes_hernan.png')
# -


