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

# +
# Old ROGER 1 data
data = pd.read_csv(DATA_PATH + '/highMass_trainset_roger1.csv', sep = ' ')
data = np.asarray(data)

data[np.where(data[:,2] == 'CL')[0], 2] = 0
data[np.where(data[:,2] == 'RIN')[0], 2] = 1
data[np.where(data[:,2] == 'BS')[0], 2] = 2
data[np.where(data[:,2] == 'IN')[0], 2] = 3
data[np.where(data[:,2] == 'ITL')[0], 2] = 4
data = data.astype('float64')

cl  = data[np.where(data[:,2] == 0)[0]]
rin = data[np.where(data[:,2] == 1)[0]]
bs  = data[np.where(data[:,2] == 2)[0]]
inf = data[np.where(data[:,2] == 3)[0]]
itl = data[np.where(data[:,2] == 4)[0]]

print('Hay ' + str(len(cl) / len(data)) + ' cluster galaxies')
print('Hay ' + str(len(bs) / len(data)) + ' backsplash galaxies')
print('Hay ' + str(len(rin) / len(data)) + ' recent infalling galaxies')
print('Hay ' + str(len(inf) / len(data)) + ' infalling galaxies')
print('Hay ' + str(len(itl) / len(data)) + ' interlooper galaxies')

# +
# Old ROGER 1 data
data = pd.read_csv(DATA_PATH + '/highmass_testset_roger1.csv', sep = ' ')
data = np.asarray(data)

data[np.where(data[:,2] == 'CL')[0], 2] = 0
data[np.where(data[:,2] == 'RIN')[0], 2] = 1
data[np.where(data[:,2] == 'BS')[0], 2] = 2
data[np.where(data[:,2] == 'IN')[0], 2] = 3
data[np.where(data[:,2] == 'ITL')[0], 2] = 4
data = data.astype('float64')

cl  = data[np.where(data[:,2] == 0)[0]]
rin = data[np.where(data[:,2] == 1)[0]]
bs  = data[np.where(data[:,2] == 2)[0]]
inf = data[np.where(data[:,2] == 3)[0]]
itl = data[np.where(data[:,2] == 4)[0]]

print('Hay ' + str(len(cl) / len(data)) + ' cluster galaxies')
print('Hay ' + str(len(bs) / len(data)) + ' backsplash galaxies')
print('Hay ' + str(len(rin) / len(data)) + ' recent infalling galaxies')
print('Hay ' + str(len(inf) / len(data)) + ' infalling galaxies')
print('Hay ' + str(len(itl) / len(data)) + ' interlooper galaxies')
# -

# ## Custom functions

# +
# Color scheme
cl_col  = 'red'
rin_col = 'green'
bs_col  = 'orange'
inf_col = 'blue'
itl_col = 'gray'

labels = ['CL', 'BS', 'RIN', 'IN', 'ITL']
# -

# # Reading data

DATA_PATH = '../data/'

# +
# ROGER 2 data

data_train = np.loadtxt(DATA_PATH + 'chuti_sorted.dat')

# data_train[:,0] 
# data_train[:,1] = Clase 
# data_train[:,2] = log(M/Msun) 
# data_train[:,3] = r/R200 
# data_train[:,4] = v/delta v 
# -

cl_ind = np.unique(data_train[:,0])
nclusters = len(cl_ind)
print('There are ' + str(nclusters) + ' clusters')

# +
ntrain = int(0.7 * nclusters)
ntest = ntrain#nclusters - ntrain

np.random.seed(91218)
random_ind = np.random.choice(cl_ind, replace = False, size = nclusters)

cl_train_ind = random_ind[:ntrain]
cl_test_ind = random_ind[ntrain:]

gal_train_ind = np.where(np.isin(data_train[:,0], cl_train_ind) == True)[0]
gal_test_ind = np.where(np.isin(data_train[:,0], cl_test_ind) == True)[0]

gal_test_ind = np.random.choice(gal_test_ind, size = 1000)
# -

print(len(gal_train_ind))

print(np.min(data_train[gal_train_ind, 2:], axis = 0))
print(np.max(data_train[gal_train_ind, 2:], axis = 0))

# +
comments = """ 
      ROGER2 model for isolated galaxy clusters with masses
      bigger than >10^{13} M_{sun}.
    """

#Roger2 = roger.RogerModel(x_dataset = data_train[gal_train_ind, 2:], y_dataset = data_train[gal_train_ind, 1], comments=comments, 
#                          ml_models = [KNeighborsClassifier(n_neighbors=63), RandomForestClassifier(max_depth=2, random_state=0)])

Roger2 = roger.RogerModel(x_dataset = data_train[gal_train_ind, 2:], y_dataset = data_train[gal_train_ind, 1], comments=comments, 
                          ml_models = [KNeighborsClassifier(n_neighbors=63)])
# -

cl_ind = np.where(data_train[:,1] == 1)[0] 
bs_ind = np.where(data_train[:,1] == 2)[0] 
rin_ind = np.where(data_train[:,1] == 3)[0] 
in_ind = np.where(data_train[:,1] == 4)[0] 
itl_ind = np.where(data_train[:,1] == 5)[0] 

print('Hay {:.2f} cluster galaxies'.format(len(cl_ind) / len(data_train)))
print('Hay {:.2f} backsplash galaxies'.format(len(bs_ind) / len(data_train)))
print('Hay {:.2f} recent infalling galaxies'.format(len(rin_ind) / len(data_train)))
print('Hay {:.2f} infalling galaxies'.format(len(in_ind) / len(data_train)))
print('Hay {:.2f} interloper galaxies'.format(len(itl_ind) / len(data_train)))

Roger2.train(path_to_saved_model = ['../data/models/roger2_KNN.joblib'])
#Roger2.train(path_to_save = ['../data/models/roger2_KNN_new0.7.joblib'])
#Roger2.train(path_to_saved_model = ['../data/models/roger2_KNN_tiny.joblib','../data/models/roger2_RF_tiny.joblib'])

# !ls ../data/

# +
h = 0.678 # Planck  y multidark
# gama data
#data_aux = np.loadtxt(DATA_PATH + 'gal_gama_30_06_26.dat')
data_aux = np.loadtxt(DATA_PATH + 'gal_gama.dat')

# data_aux[:,0] = rp/R200
# data_aux[:,1] = |Delta V|/sigma
# data_aux[:,2] = log masa del cumulo
# data_aux[:,3] = id

# +
data = np.copy(data_aux)

data[:,0] = data_aux[:,2] * h # Tengo q multiplicar por h por q en roger las unidades estan en [Msun/h] y en gama en [Msun]
data[:,1] = data_aux[:,0] 
data[:,2] = data_aux[:,1] 
# -

data.shape

print(np.min(data[:,:-1], axis = 0))
print(np.max(data[:,:-1], axis = 0))

# # Analysis

# +
path = '../data/modelos_roger2'
niter = 10

pred_prob_list = []
for i in range(niter):
  print(i)
  roger2 = models.Roger2
  roger2.train(path_to_saved_model = [path + f'/roger2_KNN_{i}.joblib', path + f'/roger2_RF_{i}.joblib'])

  pred_prob_list.append( roger2.predict_prob(data[:,:-1], n_model = 0) )
# -

pred_prob_list = np.asarray(pred_prob_list)
pred_prob = np.mean(pred_prob_list, axis = 0)
pred_prob_sd = np.std(pred_prob_list, axis = 0)

# +
readme = '''
         Data set used for testing ROGER2. Results corresponding to the averaged of 10 KNN method.

         Columns:
         -------
         LogM: Log10 of the cluster mass. [Msun/h]
         R/R200: Galaxy radial distance to the cluster center, normalized to R200.
         V/sigma: Galaxy relative velocity to cluster center normalized to cluster velocity dispersion.
         ID: Galaxy ID.
         P_cl: Probability of being a cluster galaxy.
         P_bs: Probability of being a backsplash galaxy.
         P_rin: Probability of being a recent infaller galaxy.
         P_in: Probability of being an infalling galaxy.
         P_itl: Probability of being a iterloper galaxy.
         P_cl_sd: Standard deviation of the probability of being a cluster galaxy.
         P_bs_sd: Standard deviation of the probability of being a backsplash galaxy.
         P_rin_sd: Standard deviation of the probability of being a recent infaller galaxy.
         P_in_sd: Standard deviation of the probability of being an infalling galaxy.
         P_itl_sd: Standard deviation of the probability of being a iterloper galaxy.
         '''
np.savetxt('../data/ROGER2_KNN_probabilities_gama_averaged.txt',  np.hstack((data, pred_prob, pred_prob_sd)),
          header = 'LogM R/R200 V/sigma ID P_cl P_bs P_rin P_in P_itl P_cl_sd P_bs_sd P_rin_sd P_in_sd P_itl_sd',
          comments = readme)

#pr = np.loadtxt('../data/ROGER2_KNN_probabilities_gama_full_averaged.txt', skiprows = 20)
# -

np.array_equal(pr[:,4:9], pred_prob)

pr = np.loadtxt('../data/ROGER2_KNN_probabilities_gama_averaged.txt', skiprows = 20)
class0 = np.argmax(pr[:,4:9],axis=1) + 1

pr1 = np.loadtxt('../data/ROGER2_KNN_probabilities_gama_full.txt', skiprows = 16)
class1 = np.argmax(pr1[:,5:10],axis=1) + 1

conf_mat = sk.metrics.confusion_matrix(class0, class1)
plot_confusion_matrix(conf_mat, show_absolute=True, show_normed=True)
plt.savefig('../graphs/conf_matrix_gama.pdf')

plt.scatter(pr[:,5], pr1[:,6])
#plt.plot([8,10],[8,10])

Roger2.ml_models

Roger2.trained

pred_class = Roger2.predict_class(data[:,:-1], n_model=0)
pred_prob = Roger2.predict_prob(data[:,:-1], n_model=0)
# +
readme = '''
         Data set used for testing ROGER2. Results corresponding to KNN method.

         Columns:
         -------
         LogM: Log10 of the cluster mass.
         R/R200: Galaxy radial distance to the cluster center, normalized to R200.
         V/sigma: Galaxy relative velocity to cluster center normalized to cluster velocity dispersion.
         ID: Galaxy ID.
         Pred_class: Predicted class with max probability.
         P_cl: Probability of being a cluster galaxy.
         P_bs: Probability of being a backsplash galaxy.
         P_rin: Probability of being a recent infaller galaxy.
         P_in: Probability of being an infalling galaxy.
         P_itl: Probability of being a iterloper galaxy.
         '''
np.savetxt('../data/ROGER2_KNN_probabilities_gama_full.txt',  np.hstack((data, pred_class.reshape(len(pred_class), 1), pred_prob)),
          header = 'LogM R/R200 V/sigma ID Pred_class P_cl P_bs P_rin P_in P_itl',
          comments = readme)

#pr = np.loadtxt('../data/ROGER2_KNN_probabilities_testset.txt', skiprows = 18)
# -
pr0 = np.hstack((data, pred_class.reshape(len(pred_class), 1), pred_prob))

prnew07 = np.hstack((data, pred_class.reshape(len(pred_class), 1), pred_prob))


pr = np.loadtxt('../data/ROGER2_KNN_probabilities_gama_30_06_26_.txt', skiprows = 16)


pr1 = np.loadtxt('../data/ROGER2_KNN_probabilities_gama.txt')

pr0.shape

prnew.shape


pr1[:,:3].shape

pr.shape

# +
aaux = pr
baux = prnew07
a = aaux[:,1:3]
b = baux[:,1:3]
dtype = np.dtype([('x', a.dtype), ('y', b.dtype)])

_, ia, ib = np.intersect1d(
    a.view(dtype),
    b.view(dtype),
    return_indices=True
)

fig,ax = plt.subplots(2,3)

ax[0,0].scatter(aaux[ia,5], baux[ib,5])
ax[0,0].plot([0,1], [0,1], color = 'red')

ax[0,1].scatter(aaux[ia,6], baux[ib,6])
ax[0,1].plot([0,1], [0,1], color = 'red')

ax[0,2].scatter(aaux[ia,7], baux[ib,7])
ax[0,2].plot([0,1], [0,1], color = 'red')

ax[1,0].scatter(aaux[ia,8], baux[ib,8])
ax[1,0].plot([0,1], [0,1], color = 'red')

ax[1,1].scatter(aaux[ia,9], baux[ib,9])
ax[1,1].plot([0,1], [0,1], color = 'red')

#plt.savefig('../graphs/comparacion_probalities.pdf')


# +
a = pr0[:,1:3]
b = pr[:,1:3]
dtype = np.dtype([('x', a.dtype), ('y', b.dtype)])

_, ia, ib = np.intersect1d(
    a.view(dtype),
    b.view(dtype),
    return_indices=True
)

fig,ax = plt.subplots(2,3)

ax[0,0].scatter(pr0[ia,4], pr[ib,4])
ax[0,1].scatter(pr0[ia,5], pr[ib,5])
ax[0,2].scatter(pr0[ia,6], pr[ib,6])
ax[1,0].scatter(pr0[ia,7], pr[ib,7])
ax[1,1].scatter(pr0[ia,8], pr[ib,8])
ax[1,2].scatter(pr0[ia,9], pr[ib,9])
# -


