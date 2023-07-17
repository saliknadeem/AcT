
import tensorflow as tf
from .Graphtools import k_adjacency, normalize_adjacency_matrix
#import graphtools
import numpy as np

def activation_factory(name):
    if name == 'relu':
        return tf.keras.layers.ReLU()
    elif name == 'leakyrelu':
        return tf.keras.layers.LeakyReLU(0.2)
    elif name == 'tanh':
        return tf.keras.layers.Activation('tanh')
    elif name == 'linear' or name is None:
        return tf.keras.layers.Activation('linear')
    else:
        raise ValueError('Not supported activation:', name)



class MLP(tf.keras.Model):
    def __init__(self, in_channels, out_channels, activation='relu', dropout=0):
        super(MLP, self).__init__()
        channels = [in_channels] + out_channels
        self.layers_list = []
        for i in range(1, len(channels)):
            if dropout > 0.001:
                self.layers_list.append(tf.keras.layers.Dropout(dropout))
            self.layers_list.append(tf.keras.layers.Conv2D(channels[i], kernel_size=(1, 1)))
            self.layers_list.append(tf.keras.layers.BatchNormalization())
            self.layers_list.append(activation_factory(activation))

    def call(self, x):
        # Input shape: (N,C,T,V)
        for layer in self.layers_list:
            x = layer(x)
        return x



class MultiScale_GraphConv(tf.keras.layers.Layer):
    def __init__(self,
                 num_scales,
                 in_channels,
                 out_channels,
                 A_binary,
                 disentangled_agg=True,
                 use_mask=True,
                 dropout=0,
                 activation='relu'):
        super(MultiScale_GraphConv, self).__init__()
        self.num_scales = num_scales

        if disentangled_agg:
            A_powers = [k_adjacency(A_binary, k, with_self=True) for k in range(num_scales)]
            A_powers = np.concatenate([normalize_adjacency_matrix(g) for g in A_powers])
        else:
            A_powers = [A_binary + np.eye(len(A_binary)) for k in range(num_scales)]
            A_powers = [normalize_adjacency_matrix(g) for g in A_powers]
            A_powers = [np.linalg.matrix_power(g, k) for k, g in enumerate(A_powers)]
            A_powers = np.concatenate(A_powers)

        self.A_powers = tf.convert_to_tensor(A_powers, dtype=tf.float32)
        self.use_mask = use_mask
        if use_mask:
            self.A_res = self.add_weight(shape=self.A_powers.shape, initializer=tf.random_uniform_initializer(-1e-6, 1e-6))

        self.mlp = MLP(in_channels * num_scales, [out_channels], dropout=dropout, activation=activation)

    def call(self, x):
        N, C, T, V = x.shape

        A = self.A_powers
        if self.use_mask:
            A = A + self.A_res
        support = tf.einsum('vu,nctu->nctv', A, x)
        
        support = tf.reshape(support, (-1, C, T, self.num_scales, V))
        support = tf.transpose(support, (0, 3, 1, 2, 4))
        support = tf.reshape(support, (-1, self.num_scales*C, T, V))
        out = self.mlp(support)
        return out

'''

if __name__ == "__main__":
    from graph.ntu_rgb_d import AdjMatrixGraph
    graph = AdjMatrixGraph()
    A_binary = graph.A_binary
    msgcn = MultiScale_GraphConv(num_scales=15, in_channels=3, out_channels=64, A_binary=A_binary)
    msgcn(tf.random.normal((16, 3, 30, 25)))

'''