import numpy as np


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
                 'iteration': 50000,
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