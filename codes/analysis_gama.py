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

# -

cl_ind = np.unique(data_train[:,0])
nclusters = len(cl_ind)
print('There are ' + str(nclusters) + ' clusters')

# +
ntrain = int(0.8 * nclusters)
ntest = ntrain#nclusters - ntrain

np.random.seed(91218)
random_ind = np.random.choice(cl_ind, replace = False, size = nclusters)

cl_train_ind = random_ind[:ntrain]
cl_test_ind = random_ind[ntrain:]

gal_train_ind = np.where(np.isin(data_train[:,0], cl_train_ind) == True)[0]
gal_test_ind = np.where(np.isin(data_train[:,0], cl_test_ind) == True)[0]

gal_test_ind = np.random.choice(gal_test_ind, size = 1000)

# +
comments = """ 
      ROGER2 model for isolated galaxy clusters with masses
      bigger than >10^{13} M_{sun}.
    """

Roger2 = roger.RogerModel(x_dataset = data_train[gal_train_ind, 2:], y_dataset = data_train[gal_train_ind, 1], comments=comments, 
                          ml_models = [KNeighborsClassifier(n_neighbors=63), RandomForestClassifier(max_depth=2, random_state=0)])
# -

Roger2.train(path_to_saved_model = ['../data/models/roger2_KNN_tiny.joblib','../data/models/roger2_RF_tiny.joblib'])

# +
# gama data
data_aux = np.loadtxt(DATA_PATH + 'gal_gama.dat')

# data_aux[:,0] = rp/R200
# data_aux[:,1] = |Delta V|/sigma
# data_aux[:,2] = log masa del cumulo

# +
data = np.copy(data_aux)

data[:,0] = data_aux[:,2] 
data[:,1] = data_aux[:,0] 
data[:,2] = data_aux[:,1] 
# -

data.shape

# # Analysis

Roger2.ml_models

Roger2.trained

pred_class = Roger2.predict_class(data, n_model=0)
pred_prob = Roger2.predict_prob(data, n_model=0)
# +
readme = '''
         Data set used for testing ROGER2. Results corresponding to KNN method.

         Columns:
         -------
         LogM: Log10 of the cluster mass.
         R/R200: Galaxy radial distance to the cluster center, normalized to R200.
         V/sigma: Galaxy relative velocity to cluster center normalized to cluster velocity dispersion.
         Pred_class: Predicted class with max probability.
         P_cl: Probability of being a cluster galaxy.
         P_bs: Probability of being a backsplash galaxy.
         P_rin: Probability of being a recent infaller galaxy.
         P_in: Probability of being an infalling galaxy.
         P_itl: Probability of being a iterloper galaxy.
         '''
np.savetxt('../data/ROGER2_KNN_probabilities_gama.txt',  np.hstack((data, pred_class.reshape(len(pred_class), 1), pred_prob)),
          header = 'LogM R/R200 V/sigma Pred_class P_cl P_bs P_rin P_in P_itl',
          comments = readme)

#pr = np.loadtxt('../data/ROGER2_KNN_probabilities_testset.txt', skiprows = 18)
# -

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
