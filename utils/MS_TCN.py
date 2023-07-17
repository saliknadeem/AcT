
import tensorflow as tf
import numpy as np
import copy


#import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TemporalConv(layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        self.pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        #print('padding', self.pad, 'kernel_size', kernel_size, 'dilation', dilation)
        
        self.out_channels = out_channels
        self.kernel_size = (kernel_size,1)
        self.stride = stride
        self.dilation = dilation
        self.padding = 'same'

    def build(self, input_shape):
        self.conv_frames = tf.keras.layers.Conv2D(self.out_channels, self.kernel_size,
                                                  strides=(self.stride, 1), padding=self.padding, data_format='channels_last')
        self.atrous_conv_frames = tf.keras.layers.Conv2D(self.out_channels, self.kernel_size,
                                                         dilation_rate=(self.dilation, 1), padding=self.padding, data_format='channels_last')
        super(TemporalConv, self).build(input_shape)

        self.bn = layers.BatchNormalization()

    def call(self, x):
        #x = tf.pad(x, [[0, 0], [self.pad, self.pad], [self.pad, self.pad], [0, 0]])
        #print('===x in temp',np.shape(x))
        x = layers.ZeroPadding2D(padding=((0, 0), (0,0) ))(x)  # Apply padding
        #print('===x with padding',np.shape(x))
        x = self.conv_frames(x)
        #print('===x after stride',np.shape(x))
        x = self.atrous_conv_frames(x)
        #print('===x after dilation',np.shape(x))
        #x = self.conv_d(x)
        x = self.bn(x)
        return x

class MultiScale_TemporalConv(keras.Model):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True,
                 residual_kernel_size=1,
                 activation='relu'):

        super(MultiScale_TemporalConv, self).__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches

        # Temporal Convolution branches
        self.branches = []
        for dilation in dilations:
            self.branches.append(
                keras.Sequential([
                    layers.Conv2D(
                        branch_channels,
                        kernel_size=1,
                        padding='valid',
                        data_format='channels_last'),
                    layers.BatchNormalization(),
                    layers.Activation(activation),
                    TemporalConv(
                        branch_channels,
                        branch_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=dilation)
                ])
            )

        # Additional Max & 1x1 branch
        self.branches.append(
            keras.Sequential([
                layers.Conv2D(branch_channels, kernel_size=1, strides=(stride,1), padding='valid' , data_format='channels_last'),
                layers.BatchNormalization(),
                layers.Activation(activation),
                layers.MaxPool2D(pool_size=(3,1), strides=(1,1), padding='same'),
                layers.BatchNormalization()
            ])
        )

        self.branches.append(
            keras.Sequential([
                layers.Conv2D(branch_channels, kernel_size=1, padding='valid', strides=(stride,1) ,data_format='channels_last'),
                layers.BatchNormalization()
            ])
        )
        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        self.act = layers.Activation(activation)

    def call(self, x):
        # Input dim: (N,C,T,V)
        #print("TCN_x",np.shape(x) )
        res = self.residual(x)
        #print("TCN_res",np.shape(res) )
        branch_outs = []
        ####print('===x',np.shape(x))
        for tempconv in self.branches:
            out = tempconv(x)
            #print('===out',np.shape(out))
            branch_outs.append(out)
        #print("TCN_out",np.shape(out) )
        out = tf.concat(branch_outs, axis=-1)
        #print("===res==",np.shape(res) )
        #print('===out==',np.shape(out))
        out += res
        out = self.act(out)
        return out