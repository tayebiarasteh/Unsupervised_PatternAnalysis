import numpy as np
import sklearn.cluster
import math
import matplotlib.pyplot as plt

LINE_PLOT = 0
SCATTER_PLOT = 1

def gaussian_2D(mu, sigma, nbr_pts=50):

    return np.random.multivariate_normal(mu, sigma, nbr_pts)


def KMeans(data, n_clusters, init='k-means++', n_init=10, max_iter=300,
           random_state=None):

    return sklearn.cluster.KMeans(n_clusters=n_clusters, init=init,
                                   n_init=n_init, max_iter=max_iter,
                                   random_state=random_state).fit(data)


def add_line_plot(plt_to_add, x_data, y_data, title=''):

    plt_to_add.plot(x_data, y_data)
    plt_to_add.set_title(title)


def add_scatter_plot(plt_to_add, x_data, y_data, title='', labels=None):

    if (labels is None):
        plt_to_add.scatter(x_data, y_data)
    else:
        plt_to_add.scatter(x_data, y_data, c=labels)
    plt_to_add.set_title(title)


def plot(datas_list, labels_list, plot_types=None, plot_titles=None):

    plt.clf()
    nbr_graphs = len(datas_list)
    if (plot_types is None):
        plot_types = [LINE_PLOT for i in range(nbr_graphs)]
    if (plot_titles is None):
        plot_titles = ['' for i in range(nbr_graphs)]
    if (nbr_graphs < 4):
        nbr_row = nbr_graphs
    elif (nbr_graphs == 4):
        nbr_row = 2
    else:
        nbr_row = 3
    nbr_col = math.ceil(nbr_graphs / nbr_row)
    axs = []
    for i, data in enumerate(datas_list) :
        print(data.shape)
        data_x, data_y = data.T
        plt_to_add = plt.subplot(nbr_row, nbr_col, i+1)
        if (plot_types[i] == LINE_PLOT):
            add_line_plot(plt_to_add, data_x, data_y, plot_titles[i])
        if (plot_types[i] == SCATTER_PLOT):
            add_scatter_plot(plt_to_add, data_x, data_y, plot_titles[i],
                             labels_list[i])

    plt.tight_layout()  # Avoiding overlapping texts (legend)
    plt.show()

if __name__ == "__main__":

    # Data creation ----------------------------------------------------
    mu = [0, 0]
    sigma = [[1, 0], [0, 100]]  # diagonal covariance
    gssn_1 = gaussian_2D(mu, sigma)
    sigma = [[4,5], [155,169]]
    gssn_2 = gaussian_2D(mu, sigma)

    gssns = [gssn_1, gssn_2]
    data = None
    data_labels = None
    for i, gssn in enumerate(gssns):
        if (data is None):
            data = gssn
        else:
            data = np.vstack((data,gssn))
        temp_labels = [i for j in range(gssn.shape[0])]
        if (data_labels is None):
            data_labels = temp_labels
        else:
            data_labels = data_labels + temp_labels


    # Running KMeans ---------------------------------------------------
    kmeans = KMeans(data, 2)


    # Plot graphs ------------------------------------------------------
    plot_titles = ["Initial data (drawn from different 2D gaussian distri)",
                   "Result of KMeans algorithm"]
    plot([data,data], labels_list=[data_labels, kmeans.labels_],
         plot_types=[SCATTER_PLOT, SCATTER_PLOT], plot_titles=plot_titles)
