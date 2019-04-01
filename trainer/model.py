#!/usr/bin/env python
"""This file contains all the model information: the training steps, the batch size and the model iself."""

import tensorflow as tf

configs = {
    "features": [-1,64,64,3],
    "mean": 0,
    "stddev": 0.08,
    "strides": [1, 1, 1, 1],
    "pool_strides": [1, 2, 2, 1],
    "ksize": [1, 2, 2, 1],
    "num_outputs": 512,
    "logit_outputs": 4,
    "prob": 0.9,
    "padding": "SAME",
    "pred_class_axis": 1,
    "pred_learn_rate": 0.001,
    "batch_size": 75,
    "training_steps": 1000
}


def get_training_steps():
    """Returns the number of batches that will be used to train your solution.
    It is recommended to change this value."""
    return configs["training_steps"]


def get_batch_size():
    """Returns the batch size that will be used by your solution.
    It is recommended to change this value."""
    return configs["batch_size"]


def getConvolutionalFilters(shape):
    """Outputs random values from a truncated normal distribution (by bounding the random variable from either below,
     above or both.)
     A tf.Variable() can be used as inputs for other Ops in the graph"""
    return tf.Variable(tf.truncated_normal(shape    = shape,
                                           mean     = configs["mean"],
                                           stddev   = configs["stddev"]))


def getShape(features):
    """This operation returns a tensor that has the same values as tensor with shape shape.
    If one component of shape is the special value -1, the size of that dimension is computed
    so that the total size remains constant."""
    return tf.reshape(features["x"],
                      configs["features"])

def getBatchNormalization(input_layer,conv_filter):
    """Computes a 2-D convolution given 4-D input and filter tensors"""
    c = tf.nn.conv2d(input_layer,
                     conv_filter,
                     # Strides determines how much the window shifts by in each of the dimensions.
                     strides    = configs["strides"],
                     # padding = "SAME" tries to pad evenly left and right, but if the amount of columns to be added is odd,
                     # it will add the extra column to the right
                     padding    = configs["padding"])
    c = tf.nn.relu(c)
    # Performs the max pooling on the input.
    # Down-sample an input representation reducing its dimensionality and allowing for assumptions to be made about
    # features contained in the sub-regions binned.
    conv_pool = tf.nn.max_pool(c,
                               ksize    = configs["ksize"],
                               # A stride of [1, 2, 2, 1] it won't do overlapping windows.
                               strides  = configs["pool_strides"],
                               padding  = configs["padding"])
    # Normalize the inputs of each layer, in order to fight the internal covariate shift problem.
    # Force the input of every layer to have approximately the same distribution in every training step
    return tf.layers.batch_normalization(conv_pool)

def getFullyConnected(flat_layer):
    """Creates a variable called weights, representing a fully connected weight matrix, 
    which is multiplied by the inputs to produce a Tensor of hidden units."""
    prob = configs["prob"]

    fcl = tf.contrib.layers.fully_connected(inputs              = flat_layer,
                                            num_outputs         = configs["num_outputs"],
                                            activation_fn       = tf.nn.relu,
                                            weights_regularizer = tf.contrib.layers.batch_norm)

    # Dropout approximate an exponential number of models to combine them and predict the output,
    # it is used to combat overfitting in neural networks.
    # It's implemented by randomly disconnecting some neurons of the network, resulting in
    # what is called a “thinned” network.
    fcl = tf.nn.dropout(fcl, prob)
    return tf.layers.batch_normalization(fcl)


def getLogits(features):
    input_layer = getShape(features)

    bn1 = getBatchNormalization(input_layer, getConvolutionalFilters([7, 7, 3, 32]))
    bn2 = getBatchNormalization(bn1, getConvolutionalFilters([7, 7, 32, 64]))
    bn3 = getBatchNormalization(bn2, getConvolutionalFilters([7, 7, 64, 128]))

    flat_layer = tf.contrib.layers.flatten(bn3)

    fully_conn_layer = getFullyConnected(flat_layer)

    # If activation_fn is not None, it is applied to the hidden units as well.
    return tf.contrib.layers.fully_connected(inputs=fully_conn_layer,
                                             num_outputs=configs["logit_outputs"],
                                             activation_fn=None)


def solution(features, labels, mode):
    """Returns an EstimatorSpec that is constructed using the solution that you have to write below.
    
    1 - EstimatorSpec fully defines the model to be run by an Estimator.
    
        For mode == ModeKeys.TRAIN: required fields are loss and train_op.
        For mode == ModeKeys.EVAL: required field is loss.
        For mode == ModeKeys.PREDICT: required fields are predictions.
    
        predictions: Predictions Tensor or dict of Tensor.
        loss: Training loss Tensor. Must be either scalar, or with shape [1].
        train_op: Op for the training step.
    
    
    2 - Defines Logits and Predictions
        Logits are used to cget the classes with argmax and probabilities with softmax
        Predictions used to feed estimator spec.
    
    3 - Logits are obtained trough tf.contrib.layers.fully_connected
        Uses a flatten layer from a fully connected layer (from normalization of 3 layers)"""

    logits = getLogits(features)
    predictions = {
    # Search a tensor for the largest value in any dimension. The index of the value is returned.
      "classes": tf.argmax(input    = logits,
                           axis     = configs["pred_class_axis"]),
        # Computes softmax activations.
        # Takes as input a vector of K real numbers, and normalizes it into a probability distribution
        # consisting of K probabilities.
      "probabilities": tf.nn.softmax(logits,
                                     name = "softmax_tensor")
      }

    if mode == tf.estimator.ModeKeys.PREDICT:
        # TODO: return tf.estimator.EstimatorSpec with prediction values of all classes
        return tf.estimator.EstimatorSpec(mode          = mode,
                                          predictions   = predictions)


    # When not predicting, set a loss value for train and eval
    # The softmax function is often used in the final layer of a neural network-based classifier.
        # Such networks are commonly trained under a log loss (or cross-entropy) regime.
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    if mode == tf.estimator.ModeKeys.TRAIN:
        # TODO: Let the model train here
        # TODO: return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        # Optimization made with GDO
        # To find a local minimum of a function using gradient descent, one takes steps proportional to the negative
        # of the gradient (or approximate gradient) of the function at the current point
        optimizer= tf.train.AdamOptimizer(learning_rate=configs["pred_learn_rate"])
        train_op = optimizer.minimize(loss        = loss,
                                      global_step = tf.train.get_global_step())


        return tf.estimator.EstimatorSpec(mode      = mode,
                                          loss      = loss,
                                          train_op  = train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        # The classes variable below exists of an tensor that contains all the predicted classes in a batch
        # TODO: eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=classes)}
        # TODO: return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels      = labels,
                                            predictions = predictions["classes"])}

        return tf.estimator.EstimatorSpec(mode              = mode,
                                          loss              = loss,
                                          eval_metric_ops   = eval_metric_ops)



