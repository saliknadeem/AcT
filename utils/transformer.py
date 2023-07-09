# Copyright 2021 Vittorio Mazzia. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# credits to https://www.tensorflow.org/tutorials/text/transformer

import tensorflow as tf
import numpy as np
import copy

from MS_TCN import MultiScale_TemporalConv
from MS_GCN import MultiScale_GraphConv


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.
    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.
    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    dk_sqrt = tf.math.sqrt(dk)
    scaled_attention_logits = tf.divide(matmul_qk,dk_sqrt)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

def point_wise_feed_forward_network(d_model, d_ff, activation):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(d_ff, activation=activation),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, depth, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = depth

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout, activation, **kwargs):
        super(TransformerEncoderLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation

        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"

        self.depth = d_model // self.num_heads

        self.mha = MultiHeadAttention(self.d_model, self.num_heads, self.depth)
            
        self.ffn = point_wise_feed_forward_network(self.d_model, self.d_ff, self.activation)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(self.dropout)
        self.dropout2 = tf.keras.layers.Dropout(self.dropout)

	
    def call(self, x, training):
        attn_output, _ = self.mha(x, x, x, None)  # (batch_size, input_seq_len, d_model)
            
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout, activation, n_layers, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.n_layers = n_layers
        self.encoder_layers = [TransformerEncoderLayer(d_model, num_heads,
                                                       d_ff, dropout, activation) for i in range(n_layers)]
        
    def get_config(self):
        config = {
            'n_layers': self.n_layers
        }
        
        base_config = super(TransformerEncoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x):
        for i in range(self.n_layers):
            x = self.encoder_layers[i](x)

        return x

    
class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__(**kwargs)
        self.patch_size = patch_size

    def get_config(self):
        config = {
            'patch_size': self.patch_size
        }
        
        base_config = super(Patches, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
class PatchClassEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, config, n_patches, pos_emb=None, kernel_initializer='he_normal', **kwargs):
        super(PatchClassEmbedding, self).__init__(**kwargs)
        self.d_model = d_model
        self.n_tot_patches = n_patches + 1 #This is (T+1)
        self.pos_emb = pos_emb
        self.config = config
        self.batch_size=config['BATCH_SIZE']
        self.kernel_initializer = kernel_initializer
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.class_embed = self.add_weight(shape=(1, 1, self.d_model), initializer=self.kernel_initializer, name="class_token")
        self.MS_TCN = MultiScale_TemporalConv(30,30,stride=1)
        self.MS_TCN1 = MultiScale_TemporalConv(30,30,stride=1)
        #self.MS_TCN2 = MultiScale_TemporalConv(30,30, stride=2)
        
        self.dense = tf.keras.layers.Dense(self.d_model)
        #print('self.class_embed=',self.class_embed.shape.as_list())
        if self.pos_emb is not None:
            self.pos_emb = tf.convert_to_tensor(np.load(self.pos_emb))
            self.lap_position_embedding = tf.keras.layers.Embedding(input_dim=self.pos_emb.shape[0], output_dim=self.d_model)
        else:
            self.position_embedding = tf.keras.layers.Embedding(input_dim=(self.n_tot_patches), output_dim=self.d_model)

    def get_config(self):
        config = {
            'd_model': self.d_model,
            'n_tot_patches': self.n_tot_patches,
            'kernel_initializer': self.kernel_initializer
        }
        
        base_config = super(PatchClassEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        org_inputs,inputs,X_in,A_in = inputs
        #('inputs',inputs.shape.as_list())
        x =  tf.repeat(self.class_embed, tf.shape(inputs)[0], axis=0)
        #print('base x only class',x.shape.as_list())
        #print('base inputs',inputs.shape.as_list())
        x = tf.concat((x, inputs), axis=1)
        #print('class_emb+inputs',x.shape.as_list())
        #print('self.pos_emb ===',self.pos_emb)
        if self.pos_emb is None:
            positions = tf.range(start=0, limit=self.n_tot_patches, delta=1)
            pe = self.position_embedding(positions)
        else:
            #pe = tf.concat([tf.zeros((self.n_tot_patches-1, 1)), self.pos_emb], axis=1) # N*(N+1)
            pe = self.pos_emb
            pe = tf.reshape(pe, [1, -1])
            pe = self.lap_position_embedding(pe)
            #pe = tf.reshape(pe, [self.n_tot_patches, self.d_model])

        if False:
            skeleton_emb = self.gcn([X_in, A_in])
            #print('emb skeleton_emb',skeleton_emb.shape.as_list())
            # create a tensor of zeros with shape [None, 1, 64]
            zeros_tensor = tf.zeros(shape=[tf.shape(skeleton_emb)[0], 1, self.d_model], dtype=tf.float32)
            # concatenate the zeros tensor and the sk_emb tensor along axis=1
            skeleton_emb = tf.concat([zeros_tensor, skeleton_emb], axis=1)

        if True:
            #print("---------------------------------------")
            #("inputs",inputs.shape.as_list() )
            #print("org_inputs",org_inputs.shape.as_list() )
            reshaped_x = tf.reshape(org_inputs, (-1, 30, 13, 4))
            #skeleton_emb = self.MS_TCN(reshaped_x)
            #skeleton_emb = self.tcn(org_inputs)
            #print("MS_TCN embeddings",skeleton_emb.shape.as_list() )
            
            #mstcn = MultiScale_TemporalConv(288, 288)
            #x = torch.randn(32, 288, 100, 20)

            #x = np.random.randn(32, 288, 100, 20)

            #print("MS_TCN embeddings",x.shape.as_list() )  
            skeleton_emb = self.MS_TCN1(reshaped_x)
            #print("MS_TCN2 out skeleton_emb",skeleton_emb.shape.as_list() ) 
            skeleton_emb = self.MS_TCN(skeleton_emb)
            #print("MS_TCN embeddings",skeleton_emb.shape.as_list() )
            skeleton_emb = tf.reshape(skeleton_emb, (-1, 30, 13*4))
            #skeleton_emb = self.tcn(org_inputs)
            #print("MS_TCN embeddings",skeleton_emb.shape.as_list() )
            skeleton_emb = self.dense(skeleton_emb)
            #print("skeleton_emb final",skeleton_emb.shape.as_list() )
            #x = tf.repeat(self.class_embed, tf.shape(skeleton_emb)[0], axis=0)
            #x = tf.concat((x, skeleton_emb), axis=1)
            #print("TCN embeddings+class_emb",skeleton_emb.shape.as_list() )
            #print("pe",pe.shape.as_list() )
            #x = skeleton_emb
            #exit(1)
            #zeros_tensor =  tf.repeat(self.class_embed, tf.shape(inputs)[0], axis=0)
            #print("base class tensor",zeros_tensor.shape.as_list() )
            #x = tf.concat((x, inputs), axis=1)
            zeros_tensor = tf.zeros(shape=[tf.shape(skeleton_emb)[0], 1, self.d_model], dtype=tf.float32)
            #zeros_tensor = tf.Variable(tf.zeros(shape=[tf.shape(skeleton_emb)[0], 1, self.d_model], dtype=tf.float32), trainable=True)
            #skeleton_emb_shape = tf.shape(skeleton_emb)
            #shape = [skeleton_emb_shape[0], 1, self.d_model]
            #zeros_tensor = create_trainable_variable(shape)

            # Get the dynamic batch size
            #batch_size = tf.shape(skeleton_emb)[0]

            #skeleton_emb = tf.keras.layers.Concatenate()([zeros_tensor, skeleton_emb])
       
            #print("zeros_tensor",zeros_tensor.shape.as_list() )
            skeleton_emb = tf.concat([zeros_tensor, skeleton_emb], axis=1)
            #print("skeleton_emb",skeleton_emb.shape.as_list() )
        #print('emb skeleton_emb',skeleton_emb.shape.as_list())

        #print('pe ===',pe)
        #print('emb x',x.shape.as_list())
        #print('emb pe',pe.shape.as_list())
        if self.config['USE_SKELE_EMB']:
            x =  tf.repeat(self.class_embed, tf.shape(skeleton_emb)[0], axis=0)
            x = tf.concat((x, skeleton_emb), axis=1)
            #x = x + skeleton_emb
        if self.config['USE_PE']:
            x = x + pe

        #print('encoded FINAL',encoded.shape.as_list())
        #exit(1)

        return x