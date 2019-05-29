import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor


raccoon = scipy.misc.face(gray=True)

########### pdf
pdf_raccoon = raccoon.flatten()

########## CDF
cdf_raccoon = np.cumsum(pdf_raccoon)
cdf_raccoon = cdf_raccoon / cdf_raccoon[-1]

####### uniform sampling
n_samples = 100000             # number of training samples
uni = np.random.uniform(0, 1 - 1e-10, n_samples)
new_sampled_cdf = np.searchsorted(cdf_raccoon, uni)
print(new_sampled_cdf)

########## backgraound
n0_samples = n_samples
background = np.random.uniform(n0_samples)

########## weighting
new_sampled_cdf = new_sampled_cdf * (n0_samples / (n_samples + n0_samples))
background = background * (n_samples / (n_samples + n0_samples))

######### get image coordinate
firstDim = new_sampled_cdf//raccoon.shape[1]
firstDim0 = background//raccoon.shape[1]
secondDim = np.mod(new_sampled_cdf, raccoon.shape[1])
secondDim0 = np.mod(background, raccoon.shape[1])


# ######### visualize the result (mixture)
# new_sampled_image = np.zeros(raccoon.shape)
# new_sampled_image[firstDim, secondDim] = 1
#
# new_sampled_background = np.zeros(raccoon.shape)
# new_sampled_background[firstDim0, secondDim0] = 1
#
# ##########mixture
# mixtureDensity = np.hstack(new_sampled_image, new_sampled_background)
#
#
# ########### assigning regression values
# y = np.ones_like(firstDim)
# y0 = np.zeros_like(secondDim)
# y_values = np.hstack((y, y0))

########## RandomForest
# regr = RandomForestRegressor(max_depth=2, n_estimators=100)
# regr.fit(mixtureDensity, y_values)

########### ExtraTree
# regr = ExtraTreesRegressor(max_depth=2, n_estimators=100)
# regr.fit(mixtureDensity, y_values)
