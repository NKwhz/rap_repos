# -*- coding: utf-8 -*-
from keras.layers import Add, Dense, Concatenate, concatenate, multiply, Reshape, RepeatVector, Permute, add, Flatten, Lambda
from keras.engine.topology import Layer
from keras import backend as K
import numpy as np
import tensorflow as tf
# from attention_layer_old import AttentionWithContext
from attention_layer import AttentionWithContext

# attention_size = 2


# class AttLayer(Layer):
#     def __init__(self, **kwargs):
#         self.hidden_dim = attention_size
#         super(AttLayer, self).__init__(**kwargs)

#     def build(self, input_shape):
#         self.W = self.add_weight(shape=(input_shape[-1], self.hidden_dim), initializer='he_normal', trainable=True)
#         self.bw = self.add_weight(shape=(self.hidden_dim,), initializer='zero', trainable=True)
#         # self.uw = self.add_weight(shape=(self.hidden_dim,), initializer='he_normal', trainable=True)
#         self.trainable_weights = [self.W, self.bw]
#         super(AttLayer, self).build(input_shape)

#     def call(self, x, mask=None):
#         # print(K.shape(x))
#         # x_reshaped = tf.reshape(x, [K.shape(x)[0] * K.shape(x)[1], K.shape(x)[-1]])
#         # ui = K.tanh(K.dot(x_reshaped, self.W) + self.bw)
#         # intermed = K.sum(multiply([self.uw, ui]), axis=1)
#         #
#         # weights = tf.nn.softmax(tf.reshape(intermed, [K.shape(x)[0], K.shape(x)[1]]), dim=-1)
#         # weights = tf.expand_dims(weights, axis=-1)
#         #
#         # weighted_input = x * weights
#         # return K.sum(weighted_input, axis=1)

#         # x_reshaped = K.reshape(x, [K.shape(x)[0], 2, K.shape(x)[1] // 2])
#         # print(K.shape(x_reshaped))
#         att = K.softmax(K.dot(x, self.W) + self.bw)
#         att = K.reshape(K.tile(K.reshape(att, (K.shape(att)[0], K.shape(att)[1], 1)), [1, 1, 125]), (-1, 250))
#         # print('\natt\n')
#         # K.eval(x_reshaped)
#         # print(K.shape(x_reshaped))
#         # return K.reshape(K.dot(att, x_reshaped), (K.shape(x)[0], K.shape(x)[1] // 2))
#         return att

#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], input_shape[1])

class InvMul(Layer):
    def __init__(self, array, in_count, **kwargs):
        self.factor = tf.constant(value=array, dtype=tf.float32)
        self.count = in_count
        super(InvMul, self).__init__(**kwargs)

    def call(self, mat, mask=None):
        inv_mat = tf.py_func(np.linalg.pinv, [self.factor], tf.float32)
        inv_mat = tf.tile(tf.reshape(inv_mat, [1, K.shape(inv_mat)[0], K.shape(inv_mat)[1]]), [K.shape(mat)[0], 1, 1])
        mat = tf.reshape(mat, [K.shape(mat)[0], 1, K.shape(mat)[1]])
        res = tf.matmul(inv_mat, mat)
        return res
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.count, input_shape[1])

class AttLayer(Layer):
    def __init__(self, att_size, **kwargs):
        self.hidden_dim = att_size
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.hidden_dim), initializer='he_normal', trainable=True)
        self.bw = self.add_weight(shape=(self.hidden_dim,), initializer='zero', trainable=True)
        self.trainable_weights = [self.W, self.bw]
        super(AttLayer, self).build(input_shape)

    def call(self, x, mask=None):
        att = K.softmax(K.dot(x, self.W) + self.bw)
        return att

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.hidden_dim)


# class MatMul(Layer):
#     def __init__(self, left_shape, right_shape, **kwargs):
#         self.left_shape = left_shape
#         self.right_shape = right_shape
#         super(MatMul, self).__init__(**kwargs)
#
#     def call(self, mat_pair, mask=None):
#         '''
#         mat_pair: a tuple or list of the two matrixs to be dot multiplied
#         '''
#         left, right = mat_pair
#         x = tf.matmul(left, right)
#         print(K.shape(x))
#         return x
#
#     def compute_output_shape(self, input_shape):
#         print('==========================')
#         print(input_shape)
#         print('==========================')
#         return (input_shape[0][0], self.left_shape[0], self.right_shape[1])


class MatMul(Layer):
    def __init__(self, left_shape, right_shape, **kwargs):
        self.left_shape = left_shape
        self.right_shape = right_shape
        super(MatMul, self).__init__(**kwargs)

    def call(self, mat_pair, mask=None):
        '''
        mat_pair: a tuple or list of the two matrixs to be dot multiplied
        '''
        left, right = mat_pair

        left=K.expand_dims(left)
        right=K.expand_dims(right,axis=1)
        x = tf.matmul(left, right)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.left_shape[0], self.right_shape[1])



class SumLayer(Layer):
    def __init__(self, axis, **kwargs): 
        self.t_axis = axis
        super(SumLayer, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return K.sum(x, axis=self.t_axis)

    def compute_output_shape(self, input_shape):
        return tuple([input_shape[i] for i in range(len(input_shape)) if not i == self.t_axis])

class Pinv(Layer):
    def call(self, mat, mask=None):
        '''
        mat: must be a 3-D tensor
        '''
        inv_mat = tf.map_fn(lambda x: tf.py_func(np.linalg.pinv, [x], tf.float32), mat)
        return inv_mat

    def compute_output_shape(self, input_shape):
        print('==========================')
        print(input_shape)
        print('==========================')
        return (input_shape[0], input_shape[2], input_shape[1])

# def get_pho_rep(pho_0, pho_1, pho_dim):
#     # pho_0 = tf.expand_dims(pho_0, axis=1)
#     # pho_1 = tf.expand_dims(pho_1, axis=1)
#     pho = concatenate([pho_0, pho_1], axis=1)
#     # att = Dense(units=2, use_bias=True, activation='softmax')(pho)
#     # pho_stack = Reshape((2, attention_size, ))(pho)
#     # att = Permute((2, 1))(RepeatVector(attention_size)(att))
#     # x = multiply([att, pho_stack])
#     weighted = Reshape((2, 125, ))(multiply([pho, AttLayer()(pho)]))
#     x = SumLayer(1)(weighted)
#     return x

def tensor_split(x, start, end):
    '''
    Length of tensor_split(x, 0, 100) is 100.
    '''
    return x[:, start:end]

def tensor_slice(x, i):
    return x[:, i, :]

# def get_weighted(inp, dim):
#     in_count = len(inp)
#     x = concatenate(inp, axis=1)
#     att = Dense(units=in_count, activation='softmax', use_bias=True)(x)
#
#     pho = Reshape((in_count, dim, ))(x)
#     att = Reshape((1, in_count, ))(att)
#
#     x = MatMul(left_shape=(1, in_count), right_shape=(in_count, dim))([att, pho])
#     x = Reshape((dim, ))(x)
#
#     return x, att

def get_weighted(inp, dim):
    in_count = len(inp)
    result, att = AttentionWithContext(hidden_dim=200)(inp)

    return result, att

# def get_weighted(inp, dim):
#     m = MatMul((dim, 1), (1, dim))(inp)
#     result_1 = AttentionWithContext(hidden_dim=200)(m)
#     m = Permute((2, 1))(m)
#     result_2 = AttentionWithContext(hidden_dim=200)(m)
#     result = Add()([result_1, result_2])
#     a = 0
#     return result, a

def de_attention(inp, dim, out_count):
    x, att = inp
    x = Reshape((1, dim, ))(x)

    att = Reshape((1, out_count, ))(att)
    att = Pinv()(att)

    output = MatMul(left_shape=(out_count, 1), right_shape=(1, dim))([att, x])
    # output = Flatten()(output)
    return [Lambda(tensor_slice, arguments={'i': i})(output) for i in range(out_count)]

def inv_mul(mat, array, k):
    t = InvMul(array, k)(mat)
    return [Lambda(tensor_slice, arguments={'i': i})(t) for i in range(len(array[0]))]


def get_pho_rep(pho_0, pho_1, pho_dim):
    a_0 = Dense(units=pho_dim, activation='sigmoid')(pho_0)
    a_1 = Dense(units=pho_dim, activation='sigmoid')(pho_1)
    _x_0 = multiply([pho_0, a_0])
    _x_1 = multiply([pho_1, a_1])
    x = add([_x_0, _x_1])
    return x
