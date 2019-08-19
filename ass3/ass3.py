import numpy as np
import sklearn.cluster
import math
import matplotlib.pyplot as plt

LINE_PLOT = 0
SCATTER_PLOT = 1
MULTI_Y_LINE_PLOT = 2

# Data -----------------------------------------------------------------
def gaussian_2D(mu, sigma, nbr_pts=10):

    return np.random.multivariate_normal(mu, sigma, nbr_pts)


# Clustering algorithm -------------------------------------------------
def KMeans(data, n_clusters, init='k-means++', n_init=10, max_iter=300,
           random_state=None):

    return sklearn.cluster.KMeans(n_clusters=n_clusters, init=init,
                                   n_init=n_init, max_iter=max_iter,
                                   random_state=random_state).fit(data)


# Gap stat -------------------------------------------------------------
def get_Wks(data, ks, nbr_sim=20):

    # Preparing random uniform distribution ----------------------------
    data_max = np.amax(data[:,1])
    data_min = np.amin(data[:,1])
    Wks_unif = np.zeros((nbr_sim, len(ks)))
    # Running Kmeans for each k of unif --------------------------------
    for j in range(nbr_sim):
        data_unif = np.random.uniform(data_min, data_max,
                                      data.shape[0]).reshape(-1,1)
        data_unif = np.hstack((data[:,[0]], data_unif))
        W1_unif = math.log(KMeans(data_unif,1).inertia_)
        for i, k in enumerate(ks):
            if (k == 0):
                print("Don't run Kmeans with zeros cluster")
            elif (k == 1):
                Wks_unif[j][i] = 0   # obvious, but clear
            else:
                Wks_unif[j][i] = (math.log(KMeans(data_unif, k).inertia_)
                                  - W1_unif)
    std_dev_Wks = np.std(Wks_unif, axis=0) * (math.sqrt(1 + 1/nbr_sim))
    Wks_unif = np.sum(Wks_unif, axis=0) / nbr_sim
    # Running Kmeans for each k of data --------------------------------
    Wks_data = np.zeros(len(ks))
    W1_data = math.log(KMeans(data,1).inertia_)
    for i,k in enumerate(ks):
        if (k == 0):
            print("Don't run Kmeans with zeros cluster")
        elif (k == 1):
            Wks_data[i] = 0
        else:
            Wks_data[i] = math.log(KMeans(data, k).inertia_) - W1_data

    return np.vstack((Wks_data, Wks_unif)), std_dev_Wks


def gap_curve(Wks):

    gap_curve_y = []
    for i in range(len(Wks[0])):
        gap_curve_y.append(Wks[1][i] - Wks[0][i])

    return gap_curve_y


def opti_K(ks, gap_curve_y, std_dev_Wks):

    best_gap = 0
    best_K = 0
    for i in range(1,len(gap_curve_y)):
        gap = gap_curve_y[i-1] - gap_curve_y[i] - std_dev_Wks[i]
        if (best_gap < gap):
            best_gap = gap
            best_K = i-1

    return best_K


# Plot management ------------------------------------------------------
def add_line_plot(plt_to_add, data_k, data_y, label_x, label_y, title='',
                  errors=None):

    if (errors is None):
        plt_to_add.plot(data_k, data_y)
    else:
        #plt_to_add.plot(data_k, data_y)
        plt_to_add.errorbar(data_k, data_y, yerr=errors, uplims=True,
                            lolims=True, marker='o')
    if (label_x != None):
        plt_to_add.set_xlabel(label_x)
    if (label_y != None):
        plt_to_add.set_ylabel(label_y)
    plt_to_add.set_title(title)


def add_scatter_plot(plt_to_add, data_k, data_y, label_x, label_y, title='',
                     labels=None):

    if (labels is None):
        plt_to_add.scatter(data_k, data_y)
    else:
        plt_to_add.scatter(data_k, data_y, c=labels)
    if (label_x != None):
        plt_to_add.set_xlabel(label_x)
    if (label_y != None):
        plt_to_add.set_ylabel(label_y)
    plt_to_add.set_title(title)


def add_multi_y_line_plot(plt_to_add, data_x, datas_y, label_x, label_y, title,
                          plot_labels=None):

    if (plot_labels is not None):
        for i,data_y in enumerate(datas_y):
            plt_to_add.plot(data_x, data_y, label=plot_labels[i],
                            linestyle='--', marker='o')
            plt_to_add.legend(loc = "best")
    else:
        for i,data_y in enumerate(datas_y):
            plt_to_add.plot(data_x, data_y, linestyle='--', marker='o')
    if (label_x != None):
        plt_to_add.set_xlabel(label_x)
    if (label_y != None):
        plt_to_add.set_ylabel(label_y)
    plt_to_add.set_title(title)



def plot(datas_x, datas_y, labels_list, labels_x=None, labels_y=None,
         plot_types=None, plot_titles=None, plot_labels=None,
         plot_errors=None):

    plt.clf()
    nbr_graphs = len(datas_x)
    if (labels_x is None):
        labels_x = [None for i in range(nbr_graphs)]
    if (labels_y is None):
        labels_y = [None for i in range(nbr_graphs)]
    if (plot_types is None):
        plot_types = [LINE_PLOT for i in range(nbr_graphs)]
    if (plot_titles is None):
        plot_titles = ['' for i in range(nbr_graphs)]
    if (plot_labels is None):
        plot_labels = [None for i in range(nbr_graphs)]
    if (plot_errors is None):
        plot_errors = [None for i in range(nbr_graphs)]
    if (nbr_graphs < 4):
        nbr_row = nbr_graphs
    elif (nbr_graphs == 4):
        nbr_row = 2
    else:
        nbr_row = 3
    nbr_col = math.ceil(nbr_graphs / nbr_row)
    for i in range(nbr_graphs) :
        plt_to_add = plt.subplot(nbr_row, nbr_col, i+1)
        if (plot_types[i] == LINE_PLOT):
            add_line_plot(plt_to_add, datas_x[i], datas_y[i], labels_x[i],
                          labels_y[i], plot_titles[i], plot_errors[i])
        if (plot_types[i] == SCATTER_PLOT):
            add_scatter_plot(plt_to_add, datas_x[i], datas_y[i], labels_x[i],
                             labels_y[i], plot_titles[i], labels_list[i])
        if (plot_types[i] == MULTI_Y_LINE_PLOT):
            add_multi_y_line_plot(plt_to_add, datas_x[i], datas_y[i],
                                  labels_x[i], labels_y[i], plot_titles[i],
                                  plot_labels[i])

    plt.tight_layout()  # Avoiding overlapping texts (legend)
    plt.show()


if __name__ == "__main__":

    # Data creation ----------------------------------------------------
    mu = [0, 0]
    sigma = [[30, 0], [0, 10]]  # diagonal covariance
    gssn_1 = gaussian_2D(mu, sigma)
    mu = [40, 50]
    sigma = [[25,0], [0,20]]
    gssn_2 = gaussian_2D(mu, sigma)
    mu = [40, 0]
    sigma = [[40, 0], [0, 40]]
    gssn_3 = gaussian_2D(mu, sigma)
    mu = [10, 40]
    sigma = [[0,0], [0,120]]
    gssn_4 = gaussian_2D(mu, sigma)

    gssns = [gssn_1, gssn_2, gssn_3, gssn_4]
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


    # Gap stat ---------------------------------------------------------
    nbr_ks = 10
    ks = [i for i in range(1, nbr_ks+1)]
    Wks, errors_gap_curve = get_Wks(data, ks)


    gap_curve_y = gap_curve(Wks)

    K = opti_K(ks, gap_curve_y, errors_gap_curve)
    print("Best K found by the Gap statistic for K-means is: ", K)

    # Running KMeans with optimal K* -----------------------------------
    kmeans = KMeans(data, K)

    # Plot graphs ------------------------------------------------------
    plot_titles = ["Initial data (drawn from different 2D gaussian distri)",
                   "Result of KMeans algorithm", "Expected values of log(W_k)",
                   "Gap curve"]
    plot_labels = [None, None, ["W_k of initial data", "W_k of uniform data"]
                   , None]
    labels_x = ['x','x','k','k']
    labels_y = ['y','y','log(W_k) - log(W_1)', 'Gap']
    data_x, data_y = data.T
    plot_errors = [None, None, None, errors_gap_curve]
    plot([data_x, data_x, ks, ks], [data_y, data_y, Wks, gap_curve_y],
         labels_x=labels_x, labels_y=labels_y,
         labels_list=[data_labels, kmeans.labels_, None, None],
         plot_types=[SCATTER_PLOT, SCATTER_PLOT, MULTI_Y_LINE_PLOT, LINE_PLOT],
         plot_titles=plot_titles, plot_labels=plot_labels,
         plot_errors=plot_errors)
