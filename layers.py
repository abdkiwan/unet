
import tensorflow as tf

from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers.python.layers import layers as tf_layers

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

import numpy as np


def Conv2d(X, kernel_size, num_outputs, name):
    """
    Convolution layer followed by batch normalization then activation fn:
    ----------
    Args:
        X: Tensor, [1, height, width, channels]
        kernel_size: List, filter size [height, width]
        num_outputs: Integer, number of convolution filters
        name: String, scope name
        is_training: Boolean, in training mode or not
        stride_size: List, convolution stide [height, width]
        padding: String, input padding
        activation_fn: Tensor fn, activation function on output (can be None)

    Returns:
        outputs: Tensor, [1, height+-, width+-, num_outputs]
    """

    with tf.variable_scope(name):
        num_filters_in = X.get_shape()[-1].value
        kernel_shape   = [kernel_size[0], kernel_size[1], num_filters_in, num_outputs]
        stride_shape   = [1, 1, 1, 1]

        weights = tf.get_variable('weights', kernel_shape, tf.float32, xavier_initializer())
        bias    = tf.get_variable('bias', [num_outputs], tf.float32, tf.constant_initializer(0.0))
        conv    = tf.nn.conv2d(X, weights, stride_shape, padding = 'SAME')
        outputs = tf.nn.bias_add(conv, bias)
        
        outputs = tf.nn.relu(outputs)

        return outputs


def Deconv2d(X, factor, name):
    """
    Convolution Transpose upsampling layer with bilinear interpolation weights:
    ISSUE: problems with odd scaling factors
    ----------
    Args:
        X: Tensor, [1, height, width, channels]
        factor: Integer, upsampling factor
        name: String, scope name
        padding: String, input padding
        activation_fn: Tensor fn, activation function on output (can be None)

    Returns:
        outputs: Tensor, [1, height * factor, width * factor, num_filters_in]
    """

    with tf.variable_scope(name):
        stride_shape   = [1, factor, factor, 1]
        input_shape    = tf.shape(X)
        num_filters_in = X.get_shape()[-1].value
        output_shape   = tf.stack([input_shape[0], input_shape[1] * factor, input_shape[2] * factor, num_filters_in])

        weights = bilinear_upsample_weights(factor, num_filters_in)
        outputs = tf.nn.conv2d_transpose(X, weights, output_shape, stride_shape, padding = 'SAME')

        return outputs


def bilinear_upsample_weights(factor, num_outputs):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization:
    ----------
    Args:
        factor: Integer, upsampling factor
        num_outputs: Integer, number of convolution filters

    Returns:
        outputs: Tensor, [kernel_size, kernel_size, num_outputs]
    """

    kernel_size = 2 * factor - factor % 2

    weights_kernel = np.zeros((kernel_size,
                               kernel_size,
                               num_outputs,
                               num_outputs), dtype = np.float32)

    rfactor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = rfactor - 1
    else:
        center = rfactor - 0.5

    og = np.ogrid[:kernel_size, :kernel_size]
    upsample_kernel = (1 - abs(og[0] - center) / rfactor) * (1 - abs(og[1] - center) / rfactor)

    for i in range(num_outputs):
        weights_kernel[:, :, i, i] = upsample_kernel

    init = tf.constant_initializer(value = weights_kernel, dtype = tf.float32)
    weights = tf.get_variable('weights', weights_kernel.shape, tf.float32, init)

    return weights


def Maxpool(X, kernel_size, name ):
    """
    Max pooling layer:
    ----------
    Args:
        X: Tensor, [1, height, width, channels]
        kernel_size: List, filter size [height, width]
        name: String, scope name
        stride_size: List, convolution stide [height, width]
        padding: String, input padding

    Returns:
        outputs: Tensor, [1, height / kernelsize[0], width/kernelsize[1], channels]
    """

    kernel_shape = [1, kernel_size[0], kernel_size[1], 1]
    
    outputs = tf.nn.max_pool(X, ksize = kernel_shape,
            strides = kernel_shape, padding = 'SAME', name = name)

    return outputs


def Dropout(X, dropout_prob, name):
    """
    Dropout layer:
    ----------
    Args:
        inputs: Tensor, [1, height, width, channels]
        keep_prob: Float, probability of keeping this layer
        name: String, scope name

    Returns:
        outputs: Tensor, [1, height, width, channels]
    """

    return tf.nn.dropout(X, keep_prob = dropout_prob, name = name)


def Concat(X1, X2, name):
    """
    Concatente two tensors:
    ----------
    Args:
        X1: Tensor, [1, height, width, channels]
        X2: Tensor, [1, height, width, channels]
        name: String, scope name

    Returns:
        outputs: Tensor, [1, height, width, channels1 + channels2]
    """

    return tf.concat(axis=3, values=[X1, X2], name = name)