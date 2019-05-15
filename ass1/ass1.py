import scipy.misc
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import numpy as np
import scipy.interpolate as interpolate



face = scipy.misc.face(gray=True)
print(face)

####### gaussian filter

print(face.shape)
y = nd.gaussian_filter(face, sigma=3)
print(y)


##### inverse transformation sampling

n_samples = 10000   # nbr of sample to generate

face = face / face.sum()  # normalize CDF
print(face)
y = np.cumsum(y)    # generate CDF
print(y.shape)
print(y.sum())
'''
r = np.random.rand(n_samples)   # random samples from uniform distribution

inv_cdf = interpolate.interp1d(y, np.arange(y.shape[0]))
sample = inv_cdf(r)

np.empty([face.shape[0], face.shape[1]])

'''

def inverse_transform_sampling(data, n_bins=40, n_samples=1000):
    hist, bin_edges = np.histogram(data, bins=n_bins, density=True)
    cum_values = np.zeros(bin_edges.shape)  # future CDF
    cum_values[1:] = np.cumsum(hist*np.diff(bin_edges)) # generate CDF
    inv_cdf = interpolate.interp1d(cum_values, bin_edges)   # inverse of the CDF
    r = np.random.rand(n_samples)   # random samples from uniform distribution
    return inv_cdf(r)


####### Parzen Window Estimation

def parzen_window(x, h, dim):
    '''
    Implement the Parzen window, means the hypercube window
    '''
    pass

def parzen_estimation(x, h):
    '''
    Implement the Parzen window estimator for an hypercube Window
    '''
    pass









####### plot the image
plt.gray()
plt.imshow(face)
#plt.show()
