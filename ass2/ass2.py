import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor


raccoon = scipy.misc.face(gray=True)
print(raccoon.shape)

########### pdf
pdf_raccoon = raccoon.flatten()
print(pdf_raccoon.shape)

########## CDF
cdf_raccoon = np.cumsum(pdf_raccoon)
cdf_raccoon = cdf_raccoon / cdf_raccoon[-1] #normalizing cdf

####### uniform sampling
n_samples = 200000            # number of training samples
uni = np.random.uniform(0, 1 - 1e-10, n_samples)
new_sampled_cdf = np.searchsorted(cdf_raccoon, uni)
#new_sampled_cdf = pdf_raccoon[new_sampled_cdf]
print(new_sampled_cdf)

########## backgraound
n0_samples = n_samples
background = np.random.uniform(0, pdf_raccoon.shape[0], n0_samples)
print(background)

###### running trees
xy_features = 1

if (xy_features):
    ##### target assignement
    Y = np.ones(n_samples)
    Y = np.hstack((Y,np.zeros(n0_samples)))
    print(Y.shape)

    ##### samples for RandomForest
    X = np.hstack((new_sampled_cdf,background))
    print(X.reshape((-1,1)).shape)

    ########## RandomForest
    #regr = RandomForestRegressor(max_depth=60, n_estimators=80)
    regr = ExtraTreesRegressor(max_depth=60, n_estimators=80)
    regr.fit(X.reshape((-1,1)), Y)
    pixels = [[i] for i in range(pdf_raccoon.shape[0])]


else:
    ##### target assignement
    Y = np.ones(n_samples)
    Y = np.hstack((Y,np.zeros(n0_samples)))
    print(Y.shape)

    ##### samples for RandomForest
    X = np.hstack((new_sampled_cdf,background))
    print(X.reshape((-1,1)).shape)

    ########## RandomForest
    #regr = RandomForestRegressor(max_depth=60, n_estimators=80)
    regr = ExtraTreesRegressor(max_depth=60, n_estimators=80)
    regr.fit(X.reshape((-1,1)), Y)
    #pixels = 0
    #for i in range(raccoon.shape[0])
    pixels = [[i,j] for i in range(raccoon.shape[0]) for j in range(raccoon.shape[1]) ]


print('____________________________________________________')
#print(pixels)
output = regr.predict(pixels)
output =  255*output.reshape((768,1024))
#output[output<0.5] = 0
#output[output!=0] = 1


to_plot = output
plt.gray()
plt.imshow(to_plot)
plt.show()
