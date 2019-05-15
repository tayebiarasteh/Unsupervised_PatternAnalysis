import scipy.misc
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import numpy as np
import scipy.interpolate as interpolate



raccoon = scipy.misc.face(gray=True)

####### gaussian filter

smooth_raccoon = nd.gaussian_filter(raccoon, sigma=3)
#print(y.sum())


##### inverse transformation sampling

def inverse_transform_sampling(data, n_bins=1000, n_samples=1000):
    hist, bin_edges = np.histogram(data, bins=n_bins, density=True) # generate histogram
    cum_values = np.zeros(bin_edges.shape)  # future CDF
    cum_values[1:] = np.cumsum(hist*np.diff(bin_edges)) # generate CDF
    print(cum_values.sum())
    inv_cdf = interpolate.interp1d(cum_values, bin_edges)   # inverse of the CDF
    r = np.random.rand(n_samples)   # random samples from uniform distribution
    print(r.sum())
    return inv_cdf(r)

n_samples = 10000   # nbr of sample to generate
n_bins = 1000
samples = inverse_transform_sampling(smooth_raccoon, n_bins, n_samples)

###### draw samples in raccoon image
sampled_raccoon = np.zeros(smooth_raccoon.shape)

for sample in samples:
    sampled_raccoon += np.where(smooth_raccoon==int(sample) , smooth_raccoon, 0)


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
to_plot = sampled_raccoon
plt.gray()
plt.imshow(to_plot)
plt.show()
