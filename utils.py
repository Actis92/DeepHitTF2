import numpy as np
from lifelines import KaplanMeierFitter


### MASK FUNCTIONS
'''
    mask1      : To calculate LOSS_1 (log-likelihood loss)
    mask2      : To calculate LOSS_2 (ranking loss)
'''
def compute_mask1(time, label, num_Event, num_Category):
    '''
        mask1 is required to get the log-likelihood loss
        mask1 size is [N, num_Event, num_Category]
            if not censored : one element = 1 (0 elsewhere)
            if censored     : fill elements with 1 after the censoring time (for all events)
    '''
    mask = np.zeros([np.shape(time)[0], num_Event, num_Category]) # for the first loss function
    for i in range(np.shape(time)[0]):
        if label[i,0] != 0:  #not censored
            mask[i,int(label[i,0]-1),int(time[i,0])] = 1
        else: #label[i,2]==0: censored
            mask[i,:,int(time[i,0]+1):] =  1 #fill 1 until from the censoring time (to get 1 - \sum F)
    return mask


def compute_mask2(time, meas_time, num_Category):
    '''
        mask2 is required calculate the ranking loss (for pair-wise comparision)
        mask2 size is [N, num_Category].
        - For longitudinal measurements:
             1's from the last measurement to the event time (exclusive and inclusive, respectively)
             denom is not needed since comparing is done over the same denom
        - For single measurement:
             1's from start to the event time(inclusive)
    '''
    mask = np.zeros([np.shape(time)[0], num_Category]) # for the first loss function
    if np.shape(meas_time):  #lonogitudinal measurements
        for i in range(np.shape(time)[0]):
            t1 = int(meas_time[i, 0]) # last measurement time
            t2 = int(time[i, 0]) # censoring/event time
            mask[i,(t1+1):(t2+1)] = 1  #this excludes the last measurement time and includes the event time
    else:                    #single measurement
        for i in range(np.shape(time)[0]):
            t = int(time[i, 0]) # censoring/event time
            mask[i,:(t+1)] = 1  #this excludes the last measurement time and includes the event time
    return mask

def get_random_hyperparameters():
    batch_size = [32, 64, 128] #mb_size
    layers = [1,2,3,5] #number of layers
    nodes = [50, 100, 200, 300] #number of nodes

    activation = ['relu', 'elu', 'tanh'] #non-linear activation functions

    alpha = [0.1, 0.5, 1.0, 3.0, 5.0] #alpha values -> log-likelihood loss 
    beta = [0.1, 0.5, 1.0, 3.0, 5.0] #beta values -> ranking loss
    gamma = [0.1, 0.5, 1.0, 3.0, 5.0] #gamma values -> calibration loss

    parameters = {'mb_size': batch_size[np.random.randint(len(batch_size))],
                 'iteration': 5000,
                 'dropout': 0.4,
                 'lr_train': 1e-4,
                 'h_dim_shared': nodes[np.random.randint(len(nodes))],
                 'h_dim_CS': nodes[np.random.randint(len(nodes))],
                 'num_layers_shared':layers[np.random.randint(len(layers))],
                 'num_layers_CS':layers[np.random.randint(len(layers))],
                 'active_fn': activation[np.random.randint(len(activation))],
                 'alpha': alpha[np.random.randint(len(alpha))],
                 'beta': beta[np.random.randint(len(beta))],
                 'gamma': gamma[np.random.randint(len(gamma))]
                 }
    
    return parameters

def CensoringProb(Y, T):

    T = T.reshape([-1]) # (N,) - np array
    Y = Y.reshape([-1]) # (N,) - np array

    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=(Y==0).astype(int))  # censoring prob = survival probability of event "censoring"
    G = np.asarray(kmf.survival_function_.reset_index()).transpose()
    G[1, G[1, :] == 0] = G[1, G[1, :] != 0][-1]  #fill 0 with ZoH (to prevent nan values)
    
    return G

def weighted_c_index(T_train, Y_train, Prediction, T_test, Y_test, Time):
    '''
        This is a cause-specific c(t)-index
        - Prediction      : risk at Time (higher --> more risky)
        - Time_survival   : survival/censoring time
        - Death           :
            > 1: death
            > 0: censored (including death from other cause)
        - Time            : time of evaluation (time-horizon when evaluating C-index)
    '''
    G = CensoringProb(Y_train, T_train)

    N = len(Prediction)
    A = np.zeros((N,N))
    Q = np.zeros((N,N))
    N_t = np.zeros((N,N))
    Num = 0
    Den = 0
    for i in range(N):
        tmp_idx = np.where(G[0,:] >= T_test[i])[0]

        if len(tmp_idx) == 0:
            W = (1./G[1, -1])**2
        else:
            W = (1./G[1, tmp_idx[0]])**2

        A[i, np.where(T_test[i] < T_test)] = 1. * W
        Q[i, np.where(Prediction[i] > Prediction)] = 1. # give weights

        if (T_test[i]<=Time and Y_test[i]==1):
            N_t[i,:] = 1.

    Num  = np.sum(((A)*N_t)*Q)
    Den  = np.sum((A)*N_t)

    if Num == 0 and Den == 0:
        result = -1 # not able to compute c-index!
    else:
        result = float(Num/Den)

    return result