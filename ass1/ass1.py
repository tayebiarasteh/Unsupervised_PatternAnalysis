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

n_samples = 40000   # nbr of sample to generate
n_bins = 1000
samples = inverse_transform_sampling(smooth_raccoon, n_bins, n_samples)

###### draw samples in raccoon image
sampled_raccoon = np.zeros(smooth_raccoon.shape)

for sample in samples.astype(int):
#print(samples.astype(int))
#print(smooth_raccoon)
    sampled_raccoon += np.where(smooth_raccoon==sample , smooth_raccoon, 0)

####### Parzen Window Estimation

def hypercube_kernel(h, x, x_i):
    """
    Implementation of a hypercube kernel for Parzen-window estimation.

    Keyword arguments:
        h: window width
        x: point x for density estimation, 'd x 1'-dimensional numpy array
        x_i: point from training sample, 'd x 1'-dimensional numpy array

    Returns a 'd x 1'-dimensional numpy array as input for a window function.

    """
    assert (x.shape == x_i.shape), 'vectors x and x_i must have the same dimensions'
    x_vec = (x - x_i) / (h)
    return x_vec

def parzen_window_func(x_vec, h=1):
    """
    Implementation of the window function. Returns 1 if 'd x 1'-sample vector
    lies within inside the window, 0 otherwise.

    """
    for row in x_vec:
        if np.abs(row) > (1/2):
            return 0
    return 1

def parzen_estimation(x_samples, point_x, h, d, window_func, kernel_func):
    """
    Implementation of a parzen-window estimation.

    Keyword arguments:
        x_samples: A 'n x d'-dimensional numpy array, where each sample
            is stored in a separate row. (= training sample)
        point_x: point x for density estimation, 'd x 1'-dimensional numpy array
        h: window width
        d: dimensions
        window_func: a Parzen window function (phi)
        kernel_function: A hypercube or Gaussian kernel functions

    Returns the density estimate p(x).

    """
    k_n = 0
    for row in x_samples:
        x_i = kernel_func(h=h, x=point_x, x_i=row[:,np.newaxis])
        k_n += window_func(x_i, h=h)
    return (k_n / len(x_samples)) / (h**d)


### Cross-validation to be done:

def parzen_estimation_CrossValidation(x, h):
    '''

    '''
    pass









####### plot the image
to_plot = sampled_raccoon
plt.gray()
plt.imshow(to_plot)
plt.show()
