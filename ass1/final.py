import scipy.misc
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import numpy as np
import scipy.interpolate as interpolate
import copy
import scipy.signal as sig
import sklearn.model_selection as skselect


raccoon = scipy.misc.face(gray=True)

####### gaussian filter

smooth_raccoon = nd.gaussian_filter(raccoon, sigma=3)
#print(y.sum())


####### plot the image
to_plot = smooth_raccoon
plt.gray()
plt.imshow(to_plot)
#plt.show()


########### pdf
pdf_raccoon = smooth_raccoon.flatten()

########## CDF
cdf_raccoon = np.cumsum(pdf_raccoon)
cdf_raccoon = cdf_raccoon / cdf_raccoon[-1]

####### extract with random uniform
n_samples = 100000
uni = np.random.uniform(0, 1 - 1e-10, n_samples)
new_sampled_cdf = np.searchsorted(cdf_raccoon, uni)
print(new_sampled_cdf)

######### get image coordinate

x_coord = new_sampled_cdf//raccoon.shape[1]
y_coord = np.mod(new_sampled_cdf, raccoon.shape[1])

######### visualize the result

visual_sampled = np.zeros(raccoon.shape)
visual_sampled[x_coord, y_coord] = 1

to_plot = visual_sampled
plt.gray()
plt.imshow(to_plot)
plt.show()

########## Parzen window and MLE

h_values = [5,10,15,20,25,30]
all_cost = []
#for ind, sample in enumerate(new_sampled_cdf):
#print('asonethu',sample)
for h in h_values:
    conv = np.ones((h, h))
    conv = conv / (h**2)


    kf = skselect.KFold(n_splits=5, random_state=42, shuffle=False)
    for train_index, test_index in kf.split(new_sampled_cdf):
        x_train, x_test = new_sampled_cdf[train_index], new_sampled_cdf[test_index]

        #get coord only for the train set
        x_coord = x_train//raccoon.shape[1]
        y_coord = np.mod(x_train, raccoon.shape[1])

        #Parzen window by convolution
        pdf_parzen = sig.convolve2d(visual_sampled,conv)

        #MLE func

        cost = np.log([pdf_parzen[0, i] + 1e-10 for i in range(pdf_parzen.shape[1])])
        all_cost.append(-cost.sum())

besth = np.argmax(all_cost)
print(best)
