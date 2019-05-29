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
cdf_raccoon = cdf_raccoon / cdf_raccoon[-1]

####### uniform sampling
n_samples = 100000            # number of training samples
uni = np.random.uniform(0, 1 - 1e-10, n_samples)
print(cdf_raccoon)
new_sampled_cdf = np.searchsorted(cdf_raccoon, uni)
#new_sampled_cdf = pdf_raccoon[new_sampled_cdf]
print(new_sampled_cdf)

########## backgraound
n0_samples = n_samples
background = np.random.uniform(0, pdf_raccoon.shape[0], n0_samples)
print(background)
########## weighting
#new_sampled_cdf = new_sampled_cdf * (n0_samples / (n_samples + n0_samples))
#background = background * (n_samples / (n_samples + n0_samples))

######### get image coordinate
#firstDim = new_sampled_cdf//raccoon.shape[1]
#firstDim0 = background//raccoon.shape[1]
#secondDim = np.mod(new_sampled_cdf, raccoon.shape[1])
#secondDim0 = np.mod(background, raccoon.shape[1])


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


##### target assignement
Y = np.ones(n_samples)
Y = np.hstack((Y,np.zeros(n0_samples)))
print(Y.shape)

##### samples for RandomForest
X = np.hstack((new_sampled_cdf,background))
print(X.reshape((-1,1)).shape)

########## RandomForest
regr = RandomForestRegressor(max_depth=60, n_estimators=80)
regr.fit(X.reshape((-1,1)), Y)

print('____________________________________________________')
pixels = [[i] for i in range(pdf_raccoon.shape[0])]
#print(pixels)
output = regr.predict(pixels)
output =  255*output.reshape((768,1024))
#output[output<0.5] = 0
#output[output!=0] = 1


#output = output.astype(int)
print(output)

######### get image coordinate

#x_coord = output//raccoon.shape[1]
#y_coord = np.mod(new_sampled_cdf, raccoon.shape[1])

######### visualize the result

#visual_sampled = np.zeros(raccoon.shape)
#visual_sampled[output] = 1

to_plot = output
plt.gray()
plt.imshow(to_plot)
plt.show()

########### ExtraTree
# regr = ExtraTreesRegressor(max_depth=2, n_estimators=100)
# regr.fit(mixtureDensity, y_values)
