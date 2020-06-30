import tensorflow as tf
import numpy as np


# Note that this will apply 'softmax' to the logits.
def loss_log_likelihood(pred, mask, event):
    I_1 = tf.math.sign(event)
    I_1 = tf.dtypes.cast(I_1, tf.float64)
    tmp1 = tf.math.reduce_sum(tf.math.reduce_sum(mask * pred, axis=2), axis=1, keepdims=True)
    tmp1 = tf.math.multiply(I_1, tf.math.log(tmp1))

    #for censored: log \sum P(T>t|x)
    tmp2 = tf.math.reduce_sum(tf.math.reduce_sum(mask * pred, axis=2), axis=1, keepdims=True)
    tmp2 = (1. - I_1) * tf.math.log(tmp2)

    return - tf.math.reduce_mean(tmp1 + 1.0*tmp2)

# Accuracy metric.
def loss_ranking(pred, mask, time, event, num_event, num_category):
    sigma1 = tf.constant(0.1, dtype=tf.float64)
    time  = tf.cast(time, dtype=tf.float64)
    event = tf.cast(event, dtype=tf.float64)
    eta = []
    for e in range(num_event):
        one_vector = tf.ones_like(time, dtype=tf.float64)
        I_2 = tf.cast(tf.math.equal(event, e+1), dtype = tf.float64) #indicator for event
        I_2 = tf.linalg.diag(tf.squeeze(I_2))
        tmp_e = tf.reshape(tf.slice(pred, [0, e, 0], [-1, 1, -1]), [-1, num_category]) #event specific joint prob.

        R = tf.linalg.matmul(tmp_e, tf.transpose(mask)) #no need to divide by each individual dominator
        # r_{ij} = risk of i-th pat based on j-th time-condition (last meas. time ~ event time) , i.e. r_i(T_{j})

        diag_R = tf.reshape(tf.linalg.diag_part(R), [-1, 1])
        R = tf.linalg.matmul(one_vector, tf.transpose(diag_R)) - R # R_{ij} = r_{j}(T_{j}) - r_{i}(T_{j})
        R = tf.transpose(R)                                 # Now, R_{ij} (i-th row j-th column) = r_{i}(T_{i}) - r_{j}(T_{i})

        T = tf.nn.relu(tf.math.sign(tf.linalg.matmul(one_vector, tf.transpose(time)) - 
                               tf.linalg.matmul(time, tf.transpose(one_vector))))
        # T_{ij}=1 if t_i < t_j  and T_{ij}=0 if t_i >= t_j

        T = tf.linalg.matmul(I_2, T) # only remains T_{ij}=1 when event occured for subject i

        tmp_eta = tf.math.reduce_mean(T * tf.math.exp(-R/sigma1), axis=1, keepdims=True)

        eta.append(tmp_eta)
    eta = tf.stack(eta, axis=1) #stack referenced on subjects
    eta = tf.math.reduce_mean(tf.reshape(eta, [-1, num_event]), axis=1, keepdims=True)

    return tf.math.reduce_sum(eta)

def loss_calibration(pred, mask, time, event, num_event, num_category):
    eta = []
    for e in range(num_event):
        one_vector = tf.ones_like(time, dtype=tf.float64)
        I_2 = tf.cast(tf.math.equal(event, e+1), dtype = tf.float64) #indicator for event
        tmp_e = tf.reshape(tf.slice(pred, [0, e, 0], [-1, 1, -1]), [-1, num_category]) #event specific joint prob.

        r = tf.math.reduce_sum(tmp_e * mask, axis=0) #no need to divide by each individual dominator
        tmp_eta = tf.math.reduce_mean((r - I_2)**2, axis=1, keepdims=True)

        eta.append(tmp_eta)
    eta = tf.stack(eta, axis=1) #stack referenced on subjects
    eta = tf.math.reduce_mean(tf.reshape(eta, [-1, num_event]), axis=1, keepdims=True)

    return tf.math.reduce_sum(eta) #sum over num_Events