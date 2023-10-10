#%%
###############################################################################
#               Import part
###############################################################################
import numpy as np
import matplotlib.pyplot as plt


from tp_perceptron_source import (rand_gauss, rand_bi_gauss, rand_checkers,
                                  rand_clown, plot_2d, gradient,
                                  plot_gradient, frontiere,
                                  hinge_loss, gr_hinge_loss,
                                  mse_loss, gr_mse_loss)

import seaborn as sns
from matplotlib import rc
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

plt.close('all')
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Computer Modern Roman']})
params = {'figure.figsize': (15, 8),
          'text.usetex': False,
          # 'axes.labelsize': 12,
          # 'font.size': 16,
          # 'legend.fontsize': 16,
          }
plt.rcParams.update(params)

sns.set_context("poster")
sns.set_palette("colorblind")
sns.set_style("white")
sns.axes_style()


# seed initialization
np.random.seed(seed=44)

# for saving files
saving_activated = False  # True

#%%
###############################################################################
#            Data Generation: example
###############################################################################

n = 100
mu = [1., 1.]
sigmas = [1., 1.]
rand_gauss(n, mu, sigmas)


n1 = 20
n2 = 20
mu1 = [1., 1.]
mu2 = [-1., -1.]
sigmas1 = [0.9, 0.9]
sigmas2 = [0.9, 0.9]
X1, y1 = rand_bi_gauss(n1, n2, mu1, mu2, sigmas1, sigmas2)


n1 = 50
n2 = 50
sigmas1 = 1.
sigmas2 = 5.
X2, y2 = rand_clown(n1, n2, sigmas1, sigmas2)


n1 = 75
n2 = 75
sigma = 0.1
X3, y3 = rand_checkers(n1, n2, sigma)


###############################################################################
#            Displaying labeled data
###############################################################################
plt.close("all")

plt.figure(figsize=(15, 5))
plt.subplot(131)
ax = plt.gca()
plt.title('First data set')
plot_2d(X1, y1, ax)

plt.subplot(132)
ax = plt.gca()
plt.title('Second data set')
plot_2d(X2, y2, ax)

plt.subplot(133)
ax = plt.gca()
plt.title('Third data set')
plot_2d(X3, y3, ax)
plt.show()


#%%
###############################################################################
#                Perceptron example
###############################################################################

# XXX: guess good linear classifiers on previous examples.

# XXX: check predict and predict_class functions

# XXX: properties of each loss function...?


# MSE Loss:
epsilon = 0.001
niter = 10
w_ini = np.random.randn(X1.shape[1] + 1)
lfun = mse_loss
gr_lfun = gr_mse_loss

# Gradient descent manually coded:
plt.figure()
wh, costh = gradient(X1, y1, epsilon, niter, w_ini, lfun, gr_lfun,
                     stochastic=False)
plot_gradient(X1, y1, wh, costh, lfun)
plt.suptitle('MSE and batch')
plt.tight_layout()
plt.subplots_adjust(top=0.8)
plt.show()

#%%
# Gradient descent manually coded:
epsilon = 0.05

plt.figure()
plt.suptitle('MSE and stochastic')
wh_sto, costh_sto = gradient(X1, y1, epsilon, niter * len(y1), w_ini, lfun,
                             gr_lfun, stochastic=True)
plot_gradient(X1, y1, wh_sto, costh_sto, lfun)
plt.tight_layout()
plt.subplots_adjust(top=0.8)
plt.show()

#%%
# Sklearn SGD:

clf = linear_model.SGDClassifier()
clf.fit(X1, y1)

plt.figure()
ax = plt.gca()
wsgd = [clf.intercept_[0], clf.coef_[0, 0], clf.coef_[0, 1]]
frontiere(clf, X1, y1, ax, wsgd, step=200, alpha_choice=1)
plt.show()

#%%
# Hinge Loss:
epsilon = 0.01
niter = 30
std_ini = 1.
w_ini = std_ini * np.random.randn(X1.shape[1] + 1)

lfun = hinge_loss
gr_lfun = gr_hinge_loss
wh, costh = gradient(X1, y1, epsilon, niter, w_ini, lfun,
                     gr_lfun, stochastic=False)

plt.figure()
plt.suptitle('Hinge and batch')
plot_gradient(X1, y1, wh, costh, lfun)
plt.tight_layout()
plt.subplots_adjust(top=0.8)
plt.show()

plt.figure()
plt.suptitle('Hinge and stochastic')
epsilon = 0.08
niter = 100
wh_sto, costh_sto = gradient(X1, y1, epsilon, niter, w_ini, lfun,
                             gr_lfun, stochastic=True)
plot_gradient(X1, y1, wh_sto, costh_sto, lfun)
plt.tight_layout()
plt.subplots_adjust(top=0.8)
plt.show()

#%%
# Create a figure with all the boundary displayed with a
# brighter display for the newest one using alpha_choice in plot_2d
epsilon = 0.1
niter = 50

plt.figure()
ax = plt.gca()
wh_sto, costh_sto = gradient(X1, y1, epsilon, niter, w_ini, lfun,
                             gr_lfun, stochastic=True)
indexess = np.arange(0., 1., 1. / float(niter))
plot_2d(X1, y1, ax, w=wh_sto, alpha_choice=indexess)
plt.title("Evolution de la frontière de décision (SGD)")

#%%
###############################################################################
#               Perceptron for larger dimensions
###############################################################################

# QUESTION: compare with or without 2nd order features.
