from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np


class SAM:
    """
    How to use:
        @tf.function
        def train_step_SAM(images, labels):
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = loss_object(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.first_step(gradients, model.trainable_variables)

            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = loss_object(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.second_step(gradients, model.trainable_variables)

    Taken directly from:
    https://github.com/Jannoshh/simple-sam/blob/main/sam.py
    Another implementation to look at:
    colab.research.google.com/github/sayakpaul/Sharpness-Aware-Minimization-TensorFlow/blob/main/SAM.ipynb
    """

    def __init__(self, base_optimizer, rho=0.05):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        #  use the original functions of the optimizer if needed
        self.__class__ = type(base_optimizer.__class__.__name__,
                              (self.__class__, base_optimizer.__class__),
                              {})
        self.__dict__ = base_optimizer.__dict__
        self.rho = rho
        self.base_optimizer = base_optimizer
        self.e_ws = None
        self.step = tf.Variable(1, dtype=tf.int8)

    def apply_gradients(self, grads_and_vars):
        gradients, variables = zip(*grads_and_vars)
        if tf.math.equal(self.step, tf.constant(1, dtype=tf.int8)):
            self.first_step(gradients, variables)
            self.step.assign(2)
        if tf.math.equal(self.step, tf.constant(2, dtype=tf.int8)):
            self.second_step(gradients, variables)
            self.step.assign(1)

    def first_step(self, gradients, trainable_variables):
        self.e_ws = []
        grad_norm = tf.linalg.global_norm(trainable_variables)
        for i in range(len(trainable_variables)):
            e_w = gradients[i] * self.rho / (grad_norm + 1e-12)
            trainable_variables[i].assign_add(e_w)
            self.e_ws.append(e_w)

    def second_step(self, gradients, trainable_variables):
        for i in range(len(trainable_variables)):
            trainable_variables[i].assign_add(-self.e_ws[i])
        # do the actual "sharpness-aware" update
        self.base_optimizer.apply_gradients(zip(gradients, trainable_variables))


class LARS(tf.keras.optimizers.Optimizer):
    """Layer-wise Adaptive Rate Scaling for large batch training.
    Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
    I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)
    Implements the LARS learning rate scheme presented in the paper above. This
    optimizer is useful when scaling the batch size to up to 32K without
    significant performance degradation. It is recommended to use the optimizer
    in conjunction with:
        - Gradual learning rate warm-up
        - Linear learning rate scaling
        - Poly rule learning rate decay
    Note, LARS scaling is currently only enabled for dense tensors.
    Args:
        lr: A `Tensor` or floating point value. The base learning rate.
        momentum: A floating point value. Momentum hyperparameter.
        weight_decay: A floating point value. Weight decay hyperparameter.
        eeta: LARS coefficient as used in the paper. Dfault set to LARS
            coefficient from the paper. (eeta / weight_decay) determines the
            highest scaling factor in LARS.
        epsilon: Optional epsilon parameter to be set in models that have very
            small gradients. Default set to 0.0.
        nesterov: when set to True, nesterov momentum will be enabled
    """

    def __init__(self,
                 lr,
                 momentum=0.9,
                 weight_decay=0.0001,
                 eeta=0.001,
                 epsilon=0.0,
                 nesterov=False,
                 **kwargs):

        if momentum < 0.0:
            raise ValueError("momentum should be positive: %s" % momentum)
        if weight_decay < 0.0:
            raise ValueError("weight_decay is not positive: %s" % weight_decay)
        super(LARS, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.weight_decay = K.variable(weight_decay, name='weight_decay')
            self.eeta = K.variable(eeta, name='eeta')
        self.epsilon = epsilon
        self.nesterov = nesterov

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        weights = self.get_weights()
        self.updates = [K.update_add(self.iterations, 1)]
        scaled_lr = self.lr
        w_norm = K.sqrt(K.sum([K.sum(K.square(weight))
                               for weight in weights]))
        g_norm = K.sqrt(K.sum([K.sum(K.square(grad))
                               for grad in grads]))
        scaled_lr = K.switch(K.greater(w_norm * g_norm, K.zeros([1])),
                             K.expand_dims((self.eeta * w_norm /
                                            (g_norm + self.weight_decay * w_norm +
                                             self.epsilon)) * self.lr),
                             K.ones([1]) * self.lr)
        if K.backend() == 'theano':
            scaled_lr = scaled_lr[0]  # otherwise theano raise broadcasting error
        # momentum
        moments = [K.zeros(K.int_shape(param), dtype=K.dtype(param))
                   for param in params]
        self.weights = [self.iterations] + moments
        for param, grad, moment in zip(params, grads, moments):
            v0 = (moment * self.momentum)
            v1 = scaled_lr * grad  # velocity
            veloc = v0 - v1
            self.updates.append(K.update(moment, veloc))

            if self.nesterov:
                new_param = param + (veloc * self.momentum) - v1
            else:
                new_param = param + veloc

            # Apply constraints.
            if getattr(param, 'constraint', None) is not None:
                new_param = param.constraint(new_param)

            self.updates.append(K.update(param, new_param))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'weight_decay': float(K.get_value(self.weight_decay)),
                  'epsilon': self.epsilon,
                  'eeta': float(K.get_value(self.eeta)),
                  'nesterov': self.nesterov}
        base_config = super(LARS, self).get_config()
        return dict(**base_config, **config)


class WarmupLearningRateSchedule(
    tf.keras.optimizers.schedules.LearningRateSchedule):
    """Provides a variety of learning rate decay schedules with warm up."""

    def __init__(self,
                 initial_lr,
                 steps_per_epoch=None,
                 lr_decay_type='exponential',
                 decay_factor=0.97,
                 decay_epochs=2.4,
                 total_steps=None,
                 warmup_epochs=5,
                 minimal_lr=0):
        super(WarmupLearningRateSchedule, self).__init__()
        self.initial_lr = initial_lr
        self.steps_per_epoch = steps_per_epoch
        self.lr_decay_type = lr_decay_type
        self.decay_factor = decay_factor
        self.decay_epochs = decay_epochs
        self.total_steps = total_steps
        self.warmup_epochs = warmup_epochs
        self.minimal_lr = minimal_lr

    def __call__(self, step):
        if self.lr_decay_type == 'exponential':
            assert self.steps_per_epoch is not None
            decay_steps = self.steps_per_epoch * self.decay_epochs
            lr = tf.keras.optimizers.schedules.ExponentialDecay(
                self.initial_lr, decay_steps, self.decay_factor, staircase=True)(
                step)
        elif self.lr_decay_type == 'cosine':
            assert self.total_steps is not None
            lr = 0.5 * self.initial_lr * (
                    1 + tf.cos(np.pi * tf.cast(step, tf.float32) / self.total_steps))
        elif self.lr_decay_type == 'linear':
            assert self.total_steps is not None
            lr = (1.0 -
                  tf.cast(step, tf.float32) / self.total_steps) * self.initial_lr
        elif self.lr_decay_type == 'constant':
            lr = self.initial_lr
        else:
            assert False, 'Unknown lr_decay_type : %s' % self.lr_decay_type

        if self.minimal_lr:
            lr = tf.math.maximum(lr, self.minimal_lr)

        if self.warmup_epochs:
            warmup_steps = int(self.warmup_epochs * self.steps_per_epoch)
            warmup_lr = (
                    self.initial_lr * tf.cast(step, tf.float32) /
                    tf.cast(warmup_steps, tf.float32))
            lr = tf.cond(step < warmup_steps, lambda: warmup_lr, lambda: lr)

        return lr

    def get_config(self):
        return {
            'initial_lr': self.initial_lr,
            'steps_per_epoch': self.steps_per_epoch,
            'lr_decay_type': self.lr_decay_type,
            'decay_factor': self.decay_factor,
            'decay_epochs': self.decay_epochs,
            'total_steps': self.total_steps,
            'warmup_epochs': self.warmup_epochs,
            'minimal_lr': self.minimal_lr,
        }
