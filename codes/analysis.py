# # Needed libraries

# +
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn as sk
import pandas as pd
from scipy.stats import binned_statistic_2d
import seaborn as sns
#import emcee
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
data = np.loadtxt(DATA_PATH + 'chuti_sorted.dat')

# data[:,0] = nro de cumulo
# data[:,1] =  clasificacion real de la galaxia (CL = 1, BS = 2, RIN =3, IN = 4, ITL =5)
# data[:,2] = log masa del cumulo
# data[:,3] = rp/R200
# data[:,4] = |Delta V|/sigma
# -

data.shape

# +
cl  = data[np.where(data[:,1] == 1)[0]]
bs  = data[np.where(data[:,1] == 2)[0]]
rin = data[np.where(data[:,1] == 3)[0]]
inf = data[np.where(data[:,1] == 4)[0]]
itl = data[np.where(data[:,1] == 5)[0]]

print('Hay ' + str(len(cl)) + ' cluster galaxies')
print('Hay ' + str(len(rin)) + ' recent infalling galaxies')
print('Hay ' + str(len(bs)) + ' backsplash galaxies')
print('Hay ' + str(len(inf)) + ' infalling galaxies')
print('Hay ' + str(len(itl)) + ' interlooper galaxies')
# -

aux_TNG = pd.read_csv(DATA_PATH + 'data_tng300_10_09_25.dat', sep="\t")


aux_TNG

# +
#aux_TNG = np.loadtxt(DATA_PATH + 'data_tng300_clean.dat', skiprows = 1)
# aux_TNG[:,0] = nro de cumulo
# aux_TNG[:,1] =  clasificacion real de la galaxia (CL = 0, RIN = 1, BS =2, IN = 3, ITL = 4)
# aux_TNG[:,2] = rp/R200
# aux_TNG[:,3] = |Delta V|/sigma
# aux_TNG[:,4] = log masa del cumulo

#ind = np.where((aux_TNG[:,2] < 3) &
#               (aux_TNG[:,3] < 3) )[0]

#aux_TNG = aux_TNG[ind]

#aux_TNG = pd.read_csv(DATA_PATH + 'data_tng300_clean.dat', sep="\t")
#aux_TNG = np.asarray(aux_TNG)[:,:5]


aux_TNG = pd.read_csv(DATA_PATH + 'data_tng300_10_09_25.dat', sep="\t")
aux_TNG = np.asarray(aux_TNG)

data_TNG = np.copy(aux_TNG)

# Put the correct order needed for pyroger
data_TNG[:,2] = aux_TNG[:,3] # [:,2] need to be the class
data_TNG[:,3] = aux_TNG[:,2] # [:,3] need to be log mass
data_TNG[:,4] = aux_TNG[:,4] # [:,4] need to be r/r200
data_TNG[:,5] = aux_TNG[:,5] # [:,5] need to be v/sigma

# change class beacuse in pyroger class=2 is blacksplash
#ind = np.where(aux_TNG[:,3] == 2)
#data_TNG[ind,2] = 1

# change class beacuse in pyroger class=1 is cluster
#ind = np.where(aux_TNG[:,3] == 1)
#data_TNG[ind,2] = 2

data_TNG[:,3] = data_TNG[:,3]

# data_TNG[:,0] =  halo ID
# data_TNG[:,1] =  subhalo ID
# data_TNG[:,2] =  clasificacion real de la galaxia (CL = 1, BS = 2, RIN = 3, IN = 4, ITL = 5)
# data_TNG[:,3] = log masa del cumulo
# data_TNG[:,4] = rp/R200
# data_TNG[:,5] = |Delta V|/sigma
# -


np.max(data_TNG, axis = 0)

np.min(data_TNG, axis = 0)

data_TNG.shape

# +
cl_TNG  = data_TNG[np.where(data_TNG[:,2] == 1)[0]]
bs_TNG  = data_TNG[np.where(data_TNG[:,2] == 2)[0]]
rin_TNG = data_TNG[np.where(data_TNG[:,2] == 3)[0]]
inf_TNG = data_TNG[np.where(data_TNG[:,2] == 4)[0]]
itl_TNG = data_TNG[np.where(data_TNG[:,2] == 5)[0]]

print('Hay ' + str(len(cl_TNG)) + ' cluster galaxies')
print('Hay ' + str(len(rin_TNG)) + ' recent infalling galaxies')
print('Hay ' + str(len(bs_TNG)) + ' backsplash galaxies')
print('Hay ' + str(len(inf_TNG)) + ' infalling galaxies')
print('Hay ' + str(len(itl_TNG)) + ' interlooper galaxies')
# -

# ## Plots

# +
fig,ax = plt.subplots(1,5, sharex = True, sharey = True, figsize = (14,3))

ax[0].scatter(cl[:,3], cl[:,4], c = cl_col)
ax[1].scatter(bs[:,3], bs[:,4], c = bs_col, marker = '*')
ax[2].scatter(rin[:,3], rin[:,4], c = rin_col, marker = '+')
ax[3].scatter(inf[:,3], inf[:,4], c = inf_col, marker = '<')
ax[4].scatter(itl[:,3], itl[:,4], c = itl_col, marker = '>')

ax[0].set_xlabel('$r / R_{200}$', fontsize = 12)
ax[1].set_xlabel('$r / R_{200}$', fontsize = 12)
ax[2].set_xlabel('$r / R_{200}$', fontsize = 12)
ax[3].set_xlabel('$r / R_{200}$', fontsize = 12)
ax[4].set_xlabel('$r / R_{200}$', fontsize = 12)
ax[0].set_ylabel('$v / \sigma$', fontsize = 12)
# -

mpl.colormaps['Reds'](1)

# +
fig,ax = plt.subplots(1,1, sharex = True, sharey = True, figsize = (4,4))

ind = np.random.choice(np.arange(len(cl)), replace = False, size = 1000)
sns.kdeplot(x=cl[ind, 3], y=cl[ind, 4], fill=True, alpha = 0.7, cmap='Reds', levels=3, ax = ax, zorder = 4)
sns.kdeplot(x=cl[ind, 3], y=cl[ind, 4], fill=False, alpha = 0.7, color='red', levels=3, ax = ax, zorder = 4, linestyles=['--', '-', 'solid'])

ind = np.random.choice(np.arange(len(bs)), replace = False, size = 1000)
sns.kdeplot(x=bs[ind, 3], y=bs[ind, 4], fill=True, alpha = 0.7, cmap="Oranges", levels=3, ax = ax, zorder = 2)
sns.kdeplot(x=bs[ind, 3], y=bs[ind, 4], fill=False, alpha = 0.7, color="orange", levels=3, ax = ax, zorder = 6, linestyles=['--', '-', 'solid'])

ind = np.random.choice(np.arange(len(rin)), replace = False, size = 1000)
sns.kdeplot(x=rin[ind, 3], y=rin[ind, 4], fill=True, alpha = 0.7, cmap='Greens', levels=3, ax = ax, zorder = 3)
sns.kdeplot(x=rin[ind, 3], y=rin[ind, 4], fill=False, alpha = 0.7, color='green', levels=3, ax = ax, zorder = 3, linestyles=['--', '-', 'solid'])

ind = np.random.choice(np.arange(len(inf)), replace = False, size = 1000)
sns.kdeplot(x=inf[ind, 3], y=inf[ind, 4], fill=True, alpha = 0.7, cmap="Blues", levels=3, ax = ax, zorder = 1)
sns.kdeplot(x=inf[ind, 3], y=inf[ind, 4], fill=False, alpha = 0.7, color="blue", levels=3, ax = ax, zorder = 5, linestyles=['--', '-', 'solid'])

ind = np.random.choice(np.arange(len(itl)), replace = False, size = 1000)
sns.kdeplot(x=itl[ind, 3], y=itl[ind, 4], fill=True, alpha = 0.7, cmap="Greys", levels=3, ax = ax, zorder = 0)
sns.kdeplot(x=itl[ind, 3], y=itl[ind, 4], fill=False, alpha = 0.7, color="grey", levels=3, ax = ax, zorder = 0, linestyles=['--', '-', 'solid'])

ax.set_xlabel('$r / R_{200}$', fontsize = 12)
ax.set_ylabel('$v / \sigma$', fontsize = 12)
ax.set_xlim(0,3)
ax.set_ylim(0,3)
#plt.savefig('../graphs/R_V_distros.pdf')

# +
cmaps = ['Reds', 'Oranges', 'Greens', 'Blues', 'Greys'] 
colors = ['red', 'orange', 'green', 'blue', 'grey'] 

fig,ax = plt.subplots(1,1, sharex = True, sharey = True, figsize = (4,4))

for i, dat in enumerate([cl_TNG, bs_TNG, rin_TNG, inf_TNG, itl_TNG]):
    ind = np.random.choice(np.arange(len(dat)), replace = False, size = 1000)
    sns.kdeplot(x=dat[ind, 4], y=dat[ind, 5], fill=True, alpha = 0.7, cmap=cmaps[i], levels=3, ax = ax, zorder = 4)
    sns.kdeplot(x=dat[ind, 4], y=dat[ind, 5], fill=False, alpha = 0.7, color=colors[i], levels=3, ax = ax, zorder = 4, linestyles=['--', '-', 'solid'])

ax.set_xlabel('$r / R_{200}$', fontsize = 12)
ax.set_ylabel('$v / \sigma$', fontsize = 12)
ax.set_xlim(0,3)
ax.set_ylim(0,3)
#plt.savefig('../graphs/R_V_distros_TNG.pdf')

# +
fig,ax = plt.subplots(1,5, sharex = True, sharey = True, figsize = (14,3))

ind = np.random.choice(np.arange(len(cl)), replace = False, size = 1000)
sns.kdeplot(x=cl[ind, 3], y=cl[ind, 4], fill=True, cmap="Reds", levels=5, ax = ax[0])
sns.kdeplot(x=cl_TNG[:, 4], y=cl_TNG[:, 5], fill=False, cmap="Reds", levels=5, ax = ax[0])

ind = np.random.choice(np.arange(len(bs)), replace = False, size = 1000)
sns.kdeplot(x=bs[ind, 3], y=bs[ind, 4], fill=True, cmap="Oranges", levels=5, ax = ax[1])
sns.kdeplot(x=bs_TNG[:, 4], y=bs_TNG[:, 5], fill=False, cmap="Oranges", levels=5, ax = ax[1])

ind = np.random.choice(np.arange(len(rin)), replace = False, size = 1000)
sns.kdeplot(x=rin[ind, 3], y=rin[ind, 4], fill=True, cmap="Greens", levels=5, ax = ax[2])
sns.kdeplot(x=rin_TNG[:, 4], y=rin_TNG[:, 5], fill=False, cmap="Greens", levels=5, ax = ax[2])

ind = np.random.choice(np.arange(len(inf)), replace = False, size = 1000)
sns.kdeplot(x=inf[ind, 3], y=inf[ind, 4], fill=True, cmap="Blues", levels=5, ax = ax[3])
sns.kdeplot(x=inf_TNG[:, 4], y=inf_TNG[:, 5], fill=False, cmap="Blues", levels=5, ax = ax[3])

ind = np.random.choice(np.arange(len(itl)), replace = False, size = 1000)
sns.kdeplot(x=itl[ind, 3], y=itl[ind, 4], fill=True, cmap="Greys", levels=5, ax = ax[4])
sns.kdeplot(x=itl_TNG[:, 4], y=itl_TNG[:, 5], fill=False, cmap="Greys", levels=5, ax = ax[4])

ax[0].set_xlabel('$r / R_{200}$', fontsize = 12)
ax[1].set_xlabel('$r / R_{200}$', fontsize = 12)
ax[2].set_xlabel('$r / R_{200}$', fontsize = 12)
ax[3].set_xlabel('$r / R_{200}$', fontsize = 12)
ax[4].set_xlabel('$r / R_{200}$', fontsize = 12)
ax[0].set_ylabel('$v / \sigma$', fontsize = 12)

#plt.savefig('../graphs/R_V_distros_comparison.pdf')
# -

plt.hist(data[:,2])

mass_bins = np.linspace(data[:,2].min(), data[:,2].max(), 5)

# +
cl_bins  = np.digitize(cl[:,2], mass_bins)
bs_bins  = np.digitize(bs[:,2], mass_bins)
rin_bins = np.digitize(rin[:,2], mass_bins)
inf_bins  = np.digitize(inf[:,2], mass_bins)
itl_bins = np.digitize(itl[:,2], mass_bins)

fig,ax = plt.subplots(5,5, sharex = True, sharey = True, figsize = (14,14))

for i in range(5):
    ind = np.where(cl_bins ==  (i+1))[0]
    ax[i,0].scatter(cl[ind,3], cl[ind,4], c = cl_col)
    
    ind = np.where(bs_bins ==  (i+1))[0]
    ax[i,1].scatter(bs[ind,3], bs[ind,4], c = bs_col, marker = '*')
    
    ind = np.where(rin_bins ==  (i+1))[0]
    ax[i,2].scatter(rin[ind,3], rin[ind,4], c = rin_col, marker = '+')
    
    ind = np.where(inf_bins ==  (i+1))[0]
    ax[i,3].scatter(inf[ind,3], inf[ind,4], c = inf_col, marker = '<')
    
    ind = np.where(itl_bins ==  (i+1))[0]
    ax[i,4].scatter(itl[ind,3], itl[ind,4], c = itl_col, marker = '>')

ax[4,0].set_xlabel('$r / R_{200}$', fontsize = 12)
ax[4,1].set_xlabel('$r / R_{200}$', fontsize = 12)
ax[4,2].set_xlabel('$r / R_{200}$', fontsize = 12)
ax[4,3].set_xlabel('$r / R_{200}$', fontsize = 12)
ax[4,4].set_xlabel('$r / R_{200}$', fontsize = 12)
ax[4,0].set_ylabel('$v / \sigma$', fontsize = 12)
# -

mass_bins

# +
cl_bins  = np.digitize(cl[:,2], mass_bins)
bs_bins  = np.digitize(bs[:,2], mass_bins)
rin_bins = np.digitize(rin[:,2], mass_bins)
inf_bins  = np.digitize(inf[:,2], mass_bins)
itl_bins = np.digitize(itl[:,2], mass_bins)

fig,ax = plt.subplots(4,5, sharex = True, sharey = True, figsize = (14,14))

for i in range(4):
    ind = np.where(cl_bins ==  (i+1))[0]
    if len(ind) > 1000: ind = np.random.choice(ind, replace = False, size = 1000)
    sns.kdeplot(x=cl[ind, 3], y=cl[ind, 4], fill=True, cmap="Reds", levels=5, ax = ax[i,0])
    sns.kdeplot(x=cl_TNG[:, 4], y=cl_TNG[:, 5], fill=False, cmap="Reds", levels=5, ax = ax[i,0])
    
    ind = np.where(bs_bins ==  (i+1))[0]
    if len(ind) > 1000: ind = np.random.choice(ind, replace = False, size = 1000)
    sns.kdeplot(x=bs[ind, 3], y=bs[ind, 4], fill=True, cmap="Oranges", levels=5, ax = ax[i,1])
    sns.kdeplot(x=bs_TNG[:, 4], y=bs_TNG[:, 5], fill=False, cmap="Oranges", levels=5, ax = ax[i,1])
    
    ind = np.where(rin_bins ==  (i+1))[0]
    if len(ind) > 1000: ind = np.random.choice(ind, replace = False, size = 1000)
    sns.kdeplot(x=rin[ind, 3], y=rin[ind, 4], fill=True, cmap="Greens", levels=5, ax = ax[i,2])
    sns.kdeplot(x=rin_TNG[:, 4], y=rin_TNG[:, 5], fill=False, cmap="Greens", levels=5, ax = ax[i,2])
    
    ind = np.where(inf_bins ==  (i+1))[0]
    if len(ind) > 1000: ind = np.random.choice(ind, replace = False, size = 1000)
    sns.kdeplot(x=inf[ind, 3], y=inf[ind, 4], fill=True, cmap="Blues", levels=5, ax = ax[i,3])
    sns.kdeplot(x=inf_TNG[:, 4], y=inf_TNG[:, 5], fill=False, cmap="Blues", levels=5, ax = ax[i,3])
    
    ind = np.where(itl_bins ==  (i+1))[0]
    if len(ind) > 1000: ind = np.random.choice(ind, replace = False, size = 1000)
    sns.kdeplot(x=itl[ind, 3], y=itl[ind, 4], fill=True, cmap="Greys", levels=5, ax = ax[i,4])
    sns.kdeplot(x=itl_TNG[:, 4], y=itl_TNG[:, 5], fill=False, cmap="Greys", levels=5, ax = ax[i,4])

    ax[i,0].text(0.1,0.8, '{:.2f} < M < {:.2f}'.format(mass_bins[i],mass_bins[i+1]), transform = ax[i,0].transAxes)
ax[3,0].set_xlabel('$r / R_{200}$', fontsize = 12)
ax[3,1].set_xlabel('$r / R_{200}$', fontsize = 12)
ax[3,2].set_xlabel('$r / R_{200}$', fontsize = 12)
ax[3,3].set_xlabel('$r / R_{200}$', fontsize = 12)
ax[3,4].set_xlabel('$r / R_{200}$', fontsize = 12)
ax[3,0].set_ylabel('$v / \sigma$', fontsize = 12)

ax[3,0].set_xlim(0,3)
ax[3,0].set_ylim(0,3)

#plt.savefig('../graphs/R_V_distros_massbins.pdf')
# -

# # Analysis

cl_ind = np.unique(data[:,0])
nclusters = len(cl_ind)
print('There are ' + str(nclusters) + ' clusters')

# +
ntrain = int(0.8 * nclusters)
ntest = nclusters - ntrain

np.random.seed(91218)
random_ind = np.random.choice(cl_ind, replace = False, size = nclusters)

cl_train_ind = random_ind[:ntrain]
cl_test_ind = random_ind[ntrain:]
# -

gal_train_ind = np.where(np.isin(data[:,0], cl_train_ind) == True)[0]
gal_test_ind = np.where(np.isin(data[:,0], cl_test_ind) == True)[0]

gal_test_ind = np.random.choice(gal_test_ind, size = 10000)
gal_train_ind = np.random.choice(gal_train_ind, size = 100000)

len(gal_train_ind)

len(gal_test_ind)

# +
#np.savetxt('../../test_data.dat', data[gal_test_ind])

# +
comments = """ 
      ROGER2 model for isolated galaxy clusters with masses
      bigger than >10^{13} M_{sun}.
      The input must be a np.array with shape (Nobs, 3), where
      [:,0] = log10(M_{cluster} [M_{sun}])
      [:,1] = R / R_{200}
      [:,2] = |\\Delta V| / \\sigma
    """

Roger2 = roger.RogerModel(x_dataset = data[gal_train_ind, 2:], y_dataset = data[gal_train_ind, 1], comments=comments, 
                          ml_models = [KNeighborsClassifier(n_neighbors=63), RandomForestClassifier(max_depth=2, random_state=0)])
# -

Roger2.ml_models

# !ls ../data/models

#Roger2.train(path_to_saved_model = ['../data/models/roger2_KNN.joblib','../data/models/roger2_RF.joblib'])
Roger2.train(path_to_save = ['../data/models/roger2_KNN_tiny.joblib','../data/models/roger2_RF_tiny.joblib'])

Roger2.trained

# +
real_class = data[gal_test_ind, 1]

pred_class = Roger2.predict_class(data[gal_test_ind, 2:], n_model=0)
pred_prob = Roger2.predict_prob(data[gal_test_ind, 2:], n_model=0)
# +
real_class_TNG = data_TNG[:, 2]

pred_class_TNG = Roger2.predict_class(data_TNG[:, 3:], n_model=0)
pred_prob_TNG = Roger2.predict_prob(data_TNG[:, 3:], n_model=0)

# +
readme = '''
         Data set used for testing ROGER2. Results corresponding to KNN method.

         Columns:
         -------
         
         ID_gal: Galaxy number on the original dataset.
         ID_cl: Cluster ID.
         class: Real class.
         LogM: Log10 of the cluster mass.
         R/R200: Galaxy radial distance to the cluster center, normalized to R200.
         V/sigma: Galaxy relative velocity to cluster center normalized to cluster velocity dispersion.
         P_cl: Probability of being a cluster galaxy.
         P_bs: Probability of being a backsplash galaxy.
         P_rin: Probability of being a recent infaller galaxy.
         P_in: Probability of being an infalling galaxy.
         P_itl: Probability of being a iterloper galaxy.
         '''
# #%np.savetxt('../data/ROGER2_KNN_probabilities_testset.txt',  np.hstack((gal_test_ind.reshape(len(gal_test_ind),1), data[gal_test_ind], pred_prob)),
#%          header = 'ID_gal ID_cl class LogM R/R200 V/sigma P_cl P_bs P_rin P_in P_itl',
#%          comments = readme)

#pr = np.loadtxt('../data/ROGER2_KNN_probabilities_testset.txt', skiprows = 18)

# +
readme = '''
         TNG Data set used for testing ROGER2. Results corresponding to KNN method.

         Columns:
         -------
         
         ID_cl: Cluster ID.
         ID_sub: Subhalo ID.
         class: Real class.
         LogM: Log10 of the cluster mass.
         R/R200: Galaxy radial distance to the cluster center, normalized to R200.
         V/sigma: Galaxy relative velocity to cluster center normalized to cluster velocity dispersion.
         P_cl: Probability of being a cluster galaxy.
         P_bs: Probability of being a backsplash galaxy.
         P_rin: Probability of being a recent infaller galaxy.
         P_in: Probability of being an infalling galaxy.
         P_itl: Probability of being a iterloper galaxy.
         '''

pr = np.hstack((data_TNG, pred_prob_TNG))
np.savetxt('../data/ROGER2_KNN_probabilities_testset_TNG_10_09_25.txt', pr,
          header = 'ID_cl ID_sub class LogM R/R200 V/sigma P_cl P_bs P_rin P_in P_itl',
          comments = readme)
#pr = np.loadtxt('../data/ROGER2_KNN_probabilities_testset_TNG.txt', skiprows = 17)
#pr2 = np.loadtxt('../data/ROGER2_KNN_probabilities_testset_TNG_v2.txt', skiprows = 17)
#prclean = np.loadtxt('../data/ROGER2_KNN_probabilities_testset_TNG_clean.txt', skiprows = 17)
pr = np.loadtxt('../data/ROGER2_KNN_probabilities_testset_TNG_10_09_25.txt', skiprows = 18)
# -

pr = np.loadtxt('../data/ROGER2_KNN_probabilities_testset_TNG_10_09_25_ori.txt', skiprows = 18)

pr1 = np.hstack((data_TNG, pred_prob_TNG))


plt.scatter(pr[:,6], pr1[:,6])

# +
conf_mat,_ = Roger2.confusion_matrix(real_class, pred_class)

plot_confusion_matrix(conf_mat, show_absolute=True, show_normed=True, class_names=labels)

#plt.savefig('../graphs/confusionMatrix_ROGER2_KNN.pdf')
# +
conf_mat_TNG,_ = Roger2.confusion_matrix(real_class_TNG, pred_class_TNG)

plot_confusion_matrix(conf_mat_TNG, show_absolute=True, show_normed=True, class_names=labels)

#plt.savefig('../graphs/confusionMatrix_TNG_ROGER2_KNN.pdf')

# +
cmaps = ['Reds', 'Oranges', 'Greens', 'Blues', 'Greys']
colors = ['red', 'orange', 'green', 'blue', 'grey']

fig,ax = plt.subplots(1,1, sharex = True, sharey = True, figsize = (4,4))

for i in range(5):
    aux = data[gal_test_ind[np.where(pred_class == (i+1))[0]], 2:] 
    ind = np.random.choice(np.arange(len(aux)), replace = False, size = 1000)
    sns.kdeplot(x=aux[ind, 1], y=aux[ind, 2], fill=True, alpha = 0.7, cmap=cmaps[i], levels=3, ax = ax, zorder = 4)
    sns.kdeplot(x=aux[ind, 1], y=aux[ind, 2], fill=False, alpha = 0.7, color=colors[i], levels=3, ax = ax, zorder = 4, linestyles=['--', '-', 'solid'])


ax.set_xlabel('$r / R_{200}$', fontsize = 12)
ax.set_ylabel('$v / \sigma$', fontsize = 12)
ax.set_xlim(0,3)
ax.set_ylim(0,3)

# +
mean_CL_prob, x_edges, y_edges, _ = binned_statistic_2d(data[gal_test_ind,3], data[gal_test_ind,4], pred_prob[:,0], statistic='mean', bins=20)
mean_BS_prob, _, _, _ = binned_statistic_2d(data[gal_test_ind,3], data[gal_test_ind,4], pred_prob[:,1], statistic='mean', bins=[x_edges, y_edges])
mean_RIN_prob, _, _, _ = binned_statistic_2d(data[gal_test_ind,3], data[gal_test_ind,4], pred_prob[:,2], statistic='mean', bins=[x_edges, y_edges])
mean_IN_prob, _, _, _ = binned_statistic_2d(data[gal_test_ind,3], data[gal_test_ind,4], pred_prob[:,3], statistic='mean', bins=[x_edges, y_edges])
mean_ITL_prob, _, _, _ = binned_statistic_2d(data[gal_test_ind,3], data[gal_test_ind,4], pred_prob[:,4], statistic='mean', bins=[x_edges, y_edges])

X_edges, Y_edges = np.meshgrid(x_edges, y_edges)

X_centers = 0.5 * (x_edges[:-1] + x_edges[1:])  # Centros de los bins en X
Y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])  # Centros de los bins en Y
X, Y = np.meshgrid(X_centers, Y_centers)

# +
fig,ax = plt.subplots(1,5, figsize = (14,5), sharex = True, sharey = True)
plt.subplots_adjust(wspace = 0.1)

ax[0].pcolormesh(X_edges, Y_edges, mean_CL_prob.T, cmap="Reds", shading='auto', vmin = 0, vmax = 1)
contours0 = ax[0].contour(X, Y, mean_CL_prob.T, levels=[0.2, 0.5, 0.8], colors='red', linestyles = [':','--','solid'], origin='upper')
ind = np.random.choice(np.arange(len(cl)), replace = False, size = 1000)
sns.kdeplot(x=cl[ind, 3], y=cl[ind, 4], fill=False, alpha = 0.7, color="black", levels=3, ax = ax[0], zorder = 10, linestyles=['--', '-', 'solid'])
ax[0].clabel(contours0, inline=True, fontsize=10, fmt="%.2f")
ax[0].set_xlabel('$R_{proj} / R_{200}$')
ax[0].set_ylabel('$|\Delta V_{los} / \sigma|$')
ax[0].set_title('<$P_{CL}$>')

ax[1].pcolormesh(X_edges, Y_edges, mean_BS_prob.T, cmap="Oranges", shading='auto', vmin = 0, vmax = 1)
contours1 = ax[1].contour(X, Y, mean_BS_prob.T, levels=[0.2, 0.5, 0.8], colors='orange', linestyles = [':','--','solid'], origin='upper')
ind = np.random.choice(np.arange(len(bs)), replace = False, size = 1000)
sns.kdeplot(x=bs[ind, 3], y=bs[ind, 4], fill=False, alpha = 0.7, color="black", levels=3, ax = ax[1], zorder = 10, linestyles=['--', '-', 'solid'])
ax[1].clabel(contours1, inline=True, fontsize=10, fmt="%.2f")
ax[1].set_xlabel('$R_{proj} / R_{200}$')
ax[1].set_title('<$P_{BS}$>')

ax[2].pcolormesh(X_edges, Y_edges, mean_RIN_prob.T, cmap="Greens", shading='auto', vmin = 0, vmax = 1)
contours2 = ax[2].contour(X, Y, mean_RIN_prob.T, levels=[0.2, 0.5, 0.8], colors='green', linestyles = [':','--','solid'], origin='upper')
ind = np.random.choice(np.arange(len(rin)), replace = False, size = 1000)
sns.kdeplot(x=rin[ind, 3], y=rin[ind, 4], fill=False, alpha = 0.7, color="black", levels=3, ax = ax[2], zorder = 10, linestyles=['--', '-', 'solid'])
ax[2].clabel(contours2, inline=True, fontsize=10, fmt="%.2f")
ax[2].set_xlabel('$R_{proj} / R_{200}$')
ax[2].set_title('<$P_{RIN}$>')

ax[3].pcolormesh(X_edges, Y_edges, mean_IN_prob.T, cmap="Blues", shading='auto', vmin = 0, vmax = 1)
contours3 = ax[3].contour(X, Y, mean_IN_prob.T, levels=[0.2, 0.5, 0.8], colors='blue', linestyles = [':','--','solid'], origin='upper')
ind = np.random.choice(np.arange(len(inf)), replace = False, size = 1000)
sns.kdeplot(x=inf[ind, 3], y=inf[ind, 4], fill=False, alpha = 0.7, color="black", levels=3, ax = ax[3], zorder = 10, linestyles=['--', '-', 'solid'])
ax[3].clabel(contours3, inline=True, fontsize=10, fmt="%.2f")
ax[3].set_xlabel('$R_{proj} / R_{200}$')
ax[3].set_title('<$P_{IN}$>')

ax[4].pcolormesh(X_edges, Y_edges, mean_ITL_prob.T, cmap="Greys", shading='auto', vmin = 0, vmax = 1)
contours4 = ax[4].contour(X, Y, mean_ITL_prob.T, levels=[0.2, 0.5, 0.8], colors='black', linestyles = [':','--','solid'], origin='upper')
ind = np.random.choice(np.arange(len(itl)), replace = False, size = 1000)
sns.kdeplot(x=itl[ind, 3], y=itl[ind, 4], fill=False, alpha = 0.7, color="red", levels=3, ax = ax[4], zorder = 10, linestyles=['--', '-', 'solid'])
ax[4].clabel(contours4, inline=True, fontsize=10, fmt="%.2f")
ax[4].set_xlabel('$R_{proj} / R_{200}$')
ax[4].set_title('<$P_{ITL}$>')


ax[0].set_xlim(0,3)
ax[0].set_ylim(0,3)
plt.savefig('PredProbabilities_ROGER2_KNN.pdf')
# -
# ## Let's optimize the thresholds

cm, pr = Roger2.confusion_matrix(thresholds = np.array([0.5, 0.5, 0.5, 0.5, 0.5]), pred_prob = pred_prob, real_class = real_class)

plot_confusion_matrix(cm, show_absolute=True, show_normed=True, class_names=labels)


# +
predicted_labels = np.argmax(pred_prob, axis = 1)
thresholds = np.array([0.2, 0.3, 0.25, 0.15, 0.1])  # Ejemplo de thresholds para 5 clases

aux = pred_prob - thresholds

aux = aux[np.arange(len(predicted_labels)),predicted_labels]
predicted_labels[np.where(aux > 0)[0]] = predicted_labels[np.where(aux > 0)[0]] + 1
predicted_labels[np.where(aux < 0)[0]] = -1

ind = np.where(predicted_labels != -1)[0]
conf_mat = Roger2.confusion_matrix(real_class[ind], predicted_labels[ind])
conf_mat_norm = conf_mat / np.sum(conf_mat, axis = 1, keepdims=True)
np.sum((conf_mat_norm - np.identity(5))**2)


# -

# Función de log-verosimilitud (negativo de la función de costo)
def log_likelihood(theta, pred_prob, real_class, 
                   conditions = [lambda theta, pred_class, real_class: len(np.where(theta < 0)[0]) > 0,  # If some threshold is lower than 0
                                 lambda theta, pred_class, real_class: len(np.where(theta > 1)[0]) > 0,  # If some threshold is bigger than 1
                                 lambda theta, pred_class, real_class: (len(np.where(pred_class < 0 )[0]) / len(real_class)) > 0.5  # If we lose half of the galaxies
                                ]):
    try:
        thresholds = theta
        conf_matrix, pred_labels = Roger2.confusion_matrix(thresholds = theta, pred_prob = pred_prob, real_class = real_class)
        identity_matrix = np.eye(5)
        cost = np.linalg.norm(conf_matrix - identity_matrix, ord="fro")  # Norma Frobenius
    except:
        cost = np.inf
        pred_labels = np.ones_like(real_class) * -99
    for i in conditions:
        if i(theta, pred_labels, real_class): cost = np.inf
            
    return -cost  # Maximizar el negativo del costo


# +
# Configuración de emcee
n_walkers = 20  # Número de caminantes
n_dim = 5  # Número de thresholds (una por clase)
n_iter = 100  # Número de iteraciones

# Inicialización aleatoria de los thresholds (dentro del rango [0, 1])
initial_thresholds = np.random.rand(n_walkers, n_dim)

# Crear el sampler de emcee
sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_likelihood, args=(pred_prob, real_class))

# Ejecutar el MCMC
sampler.run_mcmc(initial_thresholds, n_iter, progress=True)
# -

log_likelihood(np.array([0.5,0.5,0.5,0.5,0.5]) , pred_prob, real_class)

# +
# Extraer los resultados
samples = sampler.get_chain()  # Cadena de muestras
best_thresholds = sampler.get_chain(flat=True)[np.argmax(sampler.get_log_prob(flat=True))]  # Mejor combinación

# Resultados
print("Mejores thresholds encontrados:", best_thresholds)

# +
import corner

# Obtener las muestras planas (todas las iteraciones y caminantes combinados)
flat_samples = sampler.get_chain(flat=True)

# Graficar las distribuciones posteriores
fig = corner.corner(
    flat_samples,
    labels=[f"Threshold {i+1}" for i in range(n_dim)],
    truths=best_thresholds,  # Resaltar los mejores thresholds encontrados
    show_titles=True,
)
# -

conf_mat, pred_labels = Roger2.confusion_matrix(thresholds = best_thresholds, pred_prob = pred_prob, real_class = real_class, norm = False)

len(np.where(pred_labels < 0 )[0]) / len(real_class)

plot_confusion_matrix(conf_mat, show_absolute=True, show_normed=True, class_names=labels)

# +
bines = 6
results = []#np.zeros(shape=(bines,bines,bines,bines,bines))
puntos = np.linspace(0.2, 0.9, bines)
combinaciones = list(product(puntos, puntos, puntos, puntos, puntos))

for i, comb in enumerate(combinaciones):
    if i%10 == 0: print(str(i) + ' de ' + str(len(combinaciones)))
    results.append( log_likelihood(np.array([comb[0],comb[1],comb[2],comb[3],comb[4]]) , pred_prob, real_class) )
# -

combinaciones[np.asarray(results).argmax()]

comb = combinaciones[np.asarray(results).argmax()]
log_likelihood(np.array([comb[0],comb[1],comb[2],comb[3],comb[4]]) , pred_prob, real_class)

conf_mat, pred_labels = Roger2.confusion_matrix(thresholds = np.asarray(comb), pred_prob = pred_prob, real_class = real_class, norm = False)

plot_confusion_matrix(conf_mat, show_absolute=True, show_normed=True, class_names=labels)

# ## Analysis per mass bin


# +
roger_models = []
for i in range(4):
    ind_aux = gal_train_ind[np.where( (data[gal_train_ind, 2] > mass_bins[i]) & (data[gal_train_ind, 2] < mass_bins[i+1]))[0]]
        
    comments = f""" 
          ROGER2 model for isolated galaxy clusters with masses
          between  10^{mass_bins[i]} < M < 10^{mass_bins[i+1]}.
        """
    
    roger_models.append( roger.RogerModel(x_dataset = data[ind_aux, 2:], y_dataset = data[ind_aux, 1], comments=comments, 
                          ml_models = [KNeighborsClassifier(n_neighbors=63), RandomForestClassifier(max_depth=2, random_state=0)]) )

for model in roger_models: model.train()

# +
roger_models = []
for i in range(4):
    ind_aux = gal_train_ind[np.where( (data[gal_train_ind, 2] > mass_bins[i]) & (data[gal_train_ind, 2] < mass_bins[i+1]))[0]]
        
    comments = f""" 
          ROGER2 model for isolated galaxy clusters with masses
          between  10^{mass_bins[i]} < M < 10^{mass_bins[i+1]}.
        """
    
    roger_models.append( roger.RogerModel(x_dataset = data[ind_aux, 2:], y_dataset = data[ind_aux, 1], comments=comments, 
                          ml_models = [KNeighborsClassifier(n_neighbors=63), RandomForestClassifier(max_depth=2, random_state=0)]) )

for model in roger_models: model.train()
# -

for models in roger_models:
    # Suponiendo que tienes un modelo entrenado llamado "modelo"
    for i, model in enumerate(models.ml_models):
        dump(model, f'../data/models/roger2_massbins_{i}.joblib')  

i = 0
for i in range(4):
    model = roger_models[i]
    
    ind_aux = gal_test_ind[np.where( (data[gal_test_ind, 2] > mass_bins[i]) & (data[gal_test_ind, 2] < mass_bins[i+1]))[0]]
    
    real_class = data[gal_test_ind, 1]
    
    pred_class = model.predict_class(data[gal_test_ind, 2:], n_model=0)
    conf_mat = model.confusion_matrix(real_class, pred_class)
    
    plot_confusion_matrix(conf_mat, show_absolute=True, show_normed=True, class_names=labels)
    
    plt.savefig(f'../graphs/confusionMatrix_{i}.pdf')
    plt.clf()


# # Deprecated

# Función para calcular la matriz de confusión dados los thresholds
def calculate_confusion_matrix(thresholds, pred_prob, real_class, norm = True):
    predicted_labels = np.argmax(pred_prob, axis = 1)
    aux = pred_prob - thresholds
    
    aux = aux[np.arange(len(predicted_labels)),predicted_labels]
    predicted_labels[np.where(aux > 0)[0]] = predicted_labels[np.where(aux > 0)[0]] + 1
    predicted_labels[np.where(aux < 0)[0]] = -1
    
    ind = np.where(predicted_labels != -1)[0]
    conf_mat = Roger2.confusion_matrix(real_class[ind], predicted_labels[ind])
    if norm: conf_mat = conf_mat / np.sum(conf_mat, axis = 1, keepdims=True)
    return conf_mat, predicted_labels
