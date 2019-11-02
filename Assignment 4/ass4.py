import numpy as np
from hmmlearn import hmm
import csv
import matplotlib.pyplot as plt

path_to_sig = "./pa_mobisig/"
user_folder = "user_0/"

nbr_sig_imitated = 20
imitated_file_name = "imitated_"
nbr_sig_original = 45
original_file_name = "original_"

epsilon = 1e-10

def read_csv(file_name):
    with open(file_name+'.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if (line_count == 0):
                res = np.zeros((len(row), 1))
            else:
                res = np.hstack((res, [[float(el)] for el in row]))
            line_count += 1

    return res[:,1:]


def init_data(data):
    """['x', 'y', 'timestamp', 'pressure', 'fingerarea', 'velocityx',
    'velocityy', 'accelx', 'accely', 'accelz', 'gyrox', 'gyroy', 'gyroz']
    """
    # x and y position
    new_data = data[0:2]
    # pressure
    new_data = np.vstack((new_data, data[3:4]))
    # path-tangent angle
    #path_tang = np.arctan(data[6:7]/(data[5:6]+epsilon) + epsilon)
    path_tang = np.arctan(data[1:2]/data[0:1])
    new_data = np.vstack((new_data, path_tang))
    # path velocity magnitude
    #velocity =  np.sqrt(np.square(data[5:6])**2 + np.square(data[6:7])**2)
    velocity =  np.sqrt(np.square(data[0:1])**2 + np.square(data[1:2])**2)
    new_data = np.vstack((new_data, velocity))
    # log curvature radius
    #deriv_angle = 1 / (1 + (data[6:7]/(data[5:6]+epsilon)))
    #deriv_angle *= ((data[6:7]*data[7:8] + data[5:6]*data[8:9])
    #             / (np.square(data[7:8])+epsilon))
    #temp = np.zeros(deriv_angle.shape)
    #temp = deriv_angle[deriv_angle>0]
    curv_rad = np.log(velocity/path_tang + epsilon)
    new_data = np.vstack((new_data, curv_rad))
    # total acc magnitude
    #acc_tan = 1 / ((2*velocity)+epsilon)
    #acc_tan *= 2 * (data[6:7]*data[7:8] + data[5:6]*data[8:9])
    #acc_cent = velocity * deriv_angle
    #acc_mag = np.sqrt(np.square(acc_tan)+np.square(acc_cent))
    acc_mag = np.sqrt(np.square(velocity)+np.square(velocity*path_tang))
    new_data = np.vstack((new_data, acc_mag))
    return new_data


def norm_sig(sig):
    mean = np.mean(sig, axis=1).reshape((sig.shape[0],1))
    #cov = np.diag(np.cov(sig)).reshape((-1,1))
    #cov = 1/np.sqrt(cov)
    return (sig-mean)#*cov


def init_model(model):#, obs):
    startprob = [0 for i in range(model.n_components)]
    startprob[0] = 1
    model.startprob_ = startprob
    #model.transmat_ =

def train_model(model, obs, lengths):
    # only training data
    model.fit(obs, lengths)

def get_score(model, obs):

    # all data
    return model.score(obs)

def get_sig(start, stop, path_to_sig, user_folder, file_name):
    sig = None
    lengths = []
    for i in range(start, stop):
        if (i < 9):
            nbr_str = str(0) + str(i+1)
        else:
            nbr_str = str(i+1)
        data = read_csv(path_to_sig+user_folder+file_name+nbr_str)
        data = init_data(data)
        data = norm_sig(data)
        lengths.append(data.shape[1])
        if (i == start):
            sig = data
        else:
            sig = np.hstack((sig, data))
    return sig, lengths


############### creating model
# drop 't' and 's' from init_params and params
def fit(nbr_states, nbr_mix, max_nbr_sig_training, max_nbr_sig_test, max_nbr_sig_imitated):

    signatures, lengths = get_sig(0, max_nbr_sig_training, path_to_sig, user_folder,
                                  original_file_name)
    model = hmm.GMMHMM(n_components=nbr_states, n_mix=nbr_mix, n_iter=10, covariance_type='diag', init_params='stmcw',
                       params='stmcw')
    init_model(model)

    ######## training model
    train_model(model, signatures.T, lengths)


    temp_signatures, temp_lengths = get_sig(max_nbr_sig_training,
                                            max_nbr_sig_test,
                                            path_to_sig, user_folder,
                                            original_file_name)

    signatures = np.hstack((signatures, temp_signatures))
    lengths.extend(temp_lengths)

    temp_signatures, temp_lengths = get_sig(0, max_nbr_sig_imitated,
                                            path_to_sig, user_folder,
                                            imitated_file_name)

    signatures = np.hstack((signatures, temp_signatures))
    lengths.extend(temp_lengths)


    scores = []
    ind_counter = 0
    for el in lengths:
        scores.append(get_score(model, signatures.T[ind_counter:ind_counter+el]))
        ind_counter += el

    return scores


def plot(scores, max_nbr_sig_training, max_nbr_sig_test, max_nbr_sig_imitated):
    plt.plot(np.arange(1,max_nbr_sig_training+1), scores[:max_nbr_sig_training],
             'b.', np.arange(max_nbr_sig_training+1, max_nbr_sig_test+1),
             scores[max_nbr_sig_training:max_nbr_sig_test], 'y.',
             np.arange(max_nbr_sig_test+1, max_nbr_sig_imitated+max_nbr_sig_test+1),
             scores[max_nbr_sig_test:max_nbr_sig_test+max_nbr_sig_imitated], 'r.')

    plt.show()


max_nbr_sig_training = 40
max_nbr_sig_test = 45
max_nbr_sig_imitated = 20
nbr_states = 20
nbr_mix = 10

scores = fit(nbr_states, nbr_mix, max_nbr_sig_training, max_nbr_sig_test, max_nbr_sig_imitated)


plot(scores, max_nbr_sig_training, max_nbr_sig_test, max_nbr_sig_imitated)


Ms = [1,2,4,8,16,32,64,128]
Hs = [1,2,4,8,16,32]
tuples = [(1,32), (2,8),(2,16), (2,32), (4,4), (4,8), (4,16), (8,2), (8,4),
          (8,8), (8,16), (16, 1), (16, 2), (16, 4), (16, 8), (32, 1), (32, 2),
          (32, 4), (64, 1), (64, 2), (128, 1)]
scores_rel = np.zeros((len(Hs),len(Ms)))

tabular = 0
rel_scores = []

if tabular:
    for elem in tuples:
        scores = fit(elem[0], elem[1], max_nbr_sig_training, max_nbr_sig_test, max_nbr_sig_imitated)
        score_test = np.sum(np.asarray(scores[max_nbr_sig_training:max_nbr_sig_test]))
        score_imit = np.sum(np.asarray(scores[max_nbr_sig_test:max_nbr_sig_test+max_nbr_sig_imitated]))
        rel_scores.append(score_imit/score_test)
        print('(M, H) = ', elem, ' : ', rel_scores[-1])
