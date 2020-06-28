import tensorflow as tf
from tensorflow.keras import Model, layers


class FCLayer(tf.keras.layers.Layer):
    def __init__(self, num_layers, h_dim, dropout_rate=None, activation=tf.nn.relu, kernel_regularizer=None):
        super(FCLayer, self).__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.fcn = layers.Dense(h_dim, activation=activation,
                                 kernel_initializer='glorot_uniform', kernel_regularizer=kernel_regularizer)
        self.dropout = layers.Dropout(rate=dropout_rate)
        
    def call(self, x):
        for layer in range(self.num_layers):
            x = self.fcn(x)
            if not self.keep_prob is None:
                x = self.dropout(x)
        return x

class DeepHit(Model):
    def __init__(self, num_layers_shared, h_dim_shared, activation, dropout_rate, kernel_regularizer,
                num_layers_cs, h_dim_cs, num_event, num_category):
        super(DeepHit, self).__init__()
        self.num_event = num_event
        self.num_category = num_category
        self.shared_net = FCLayer(num_layers_shared, h_dim_shared, dropout_rate, activation, kernel_regularizer)
        self.cs_net = FCLayer(num_layers_cs, h_dim_cs, dropout_rate, activation, kernel_regularizer)
        self.dropout = layers.Dropout(rate=dropout_rate)
        self.out_net = layers.Dense(num_event * num_category, activation=tf.nn.softmax,
                                 kernel_initializer='glorot_uniform', kernel_regularizer=kernel_regularizer)
        
    def call(self, inputs):
        x = self.shared_net(inputs)
        x = layers.Concatenate(axis=1)([inputs, x])
        out = []
        for _ in range(self.num_event):
            cs_out = self.cs_net(x)
            out.append(cs_out)
        out = tf.stack(out, axis=1) # stack referenced on subject
        out = tf.reshape(out, [-1, self.num_event*self.h_dim_cs])
        out = self.dropout(out)
        out = self.out_net(out)
        out = tf.reshape(out, [-1, self.num_event, self.num_category])
        return out
