import scipy.misc
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import numpy as np
import scipy.interpolate as interpolate
import copy
from sklearn.model_selection import train_test_split


raccoon = scipy.misc.face(gray=True)

####### gaussian filter

smooth_raccoon = nd.gaussian_filter(raccoon, sigma=3)
#print(y.sum())


##### inverse transformation sampling

def inverse_transform_sampling(data, n_bins=1000, n_samples=1000):
    '''
    perform inverse transform sampling with cumulative density function
    '''
    hist, bin_edges = np.histogram(data, bins=n_bins, density=True) # generate histogram
    cum_values = np.zeros(bin_edges.shape)  # future CDF
    cum_values[1:] = np.cumsum(hist*np.diff(bin_edges)) # generate CDF
    print(cum_values.sum())
    inv_cdf = interpolate.interp1d(cum_values, bin_edges)   # inverse of the CDF
    r = np.random.rand(n_samples)   # random samples from uniform distribution
    print(r.sum())
    samples = inv_cdf(r)
    sampled_raccoon = np.zeros(smooth_raccoon.shape)
    ###### draw samples in raccoon image
    for sample in samples.astype(int):
    #print(samples.astype(int))
    #print(smooth_raccoon)
        sampled_raccoon += np.where(smooth_raccoon==sample , smooth_raccoon, 0)
    return sampled_raccoon

n_samples = 10000   # nbr of sample to generate
n_bins = 1000

sampled_raccoon = inverse_transform_sampling(smooth_raccoon, n_bins, n_samples)
print('inverse_transform_sampling_done')


####### Parzen Window Estimation

def hypercube_kernel(x_samples, x, h):
    """
    Implementation of a parzen-window hypercube kernel.
    """
    nbr = 0
    for x_i in x_samples:
        if x_i[0]!=0:   #not the right color
            is_in = 1
            for coord in range(0, x_i.shape[0]):
                if (np.abs(x_i[coord] - x[coord]) >= (h/2)):
                    is_in = 0
            if is_in:
                nbr += 1
        #print('saotnhq', nbr)
    return (nbr / len(x_samples)) / (h**x_samples.shape[1])


def parzen_estimation(x_samples, h):
    """
    Implementation of a parzen-window estimator.
    """
    x_new = copy.deepcopy(x_samples)
    for i in range(0, x_samples.shape[0]):
        if x_samples[i][0]==0:     #not the right color
            x_new[i][0] = hypercube_kernel(x_samples, x_samples[i], h)
            print(x_new[i][0])
    return x_new

'''
def parzen_estimation(x_samples, h):
    """
    Implementation of a parzen-window estimator.
    """
    x_new = copy.deepcopy(x_samples)
    for i in range(0, x_samples[0]):
        for j in range(0, x_samples[1]):
            pass


    for i in range(0, x_samples):
        if x_samples[i][0]==0:     #not the right color
            x_new[i] = int(hypercube_kernel(x_samples, x_samples[i], h))
            print(x_new[i])
    return x_new
    '''


def reformat_raccoon(old):
    new = np.zeros((old.shape[0]*old.shape[1], 3))
    for i in range(0, old.shape[0]):
        for j in range(0, old.shape[1]):
            new[i*old.shape[0]+j] = np.array([old[i][j],i,j])
            #print(new)
    return new

print(sampled_raccoon.shape)
#print(reformat_raccoon(sampled_raccoon).shape, ' and ', sampled_raccoon.shape[0]*sampled_raccoon.shape[1])
sampled_raccoon = parzen_estimation(reformat_raccoon(sampled_raccoon), 500)


### Cross-validation:

def parzen_estimation_CrossValidation(x, h):
    '''
    cross validation
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)








####### plot the image
to_plot = sampled_raccoon
plt.gray()
plt.imshow(to_plot)
#plt.show()
