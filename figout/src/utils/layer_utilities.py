import tensorflow as tf
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import InputSpec
import numpy as np

from framework.submodel import Block

"""-----------------------------------   CYCLEGAN   ------------------------------------------"""


class LeastSquareLoss(tf.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.math.squared_difference(y_pred, y_true))


class ReflectionPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding, **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3]

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')


class Padding2D(tf.keras.layers.Layer):
    """Custom padding layer for cycleGAN. No trainable parameters, just image padding"""

    def __init__(self, padding, padding_mode="CONSTANT", name="Padding2D", **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        self.padding_mode = padding_mode
        super(Padding2D, self).__init__(name=name, trainable=False, **kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3]

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        # NOTE: this is for 'channel last'; for 'channel first', use: [h, h], [w, w], [0, 0], [0, 0]
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], mode=self.padding_mode)


"""-----------------------------------   ProgressiveGAN   ------------------------------------------"""


class PixelNormalization(tf.keras.layers.Layer):
    """Pixelwise feature vector normalization.
    Source: ProgressiveGANs"""

    def __init__(self, epsilon=1e-8, name="Pixelwise Normalization", name_suffix=None, **kwargs):
        if name_suffix:
            name = name + ': ' + name_suffix
        super().__init__(name=name, trainable=False, **kwargs)
        self.epsilon = epsilon

    def call(self, x, **kwargs):
        return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=3, keepdims=True) + self.epsilon)

    def compute_output_shape(self, input_shape):
        return input_shape


class MinibatchSTDDEV(tf.keras.layers.Layer):
    """From Progressive GANs.
      group_size: a integer number, minibatch must be divisible by (or smaller than) group_size.
    """

    def __init__(self, group_size=4):
        super(MinibatchSTDDEV, self).__init__()
        self.group_size = group_size

    def call(self, inputs, **kwargs):
        group_size = tf.minimum(self.group_size,
                                tf.shape(inputs)[0])  # Minibatch must be divisible by (or smaller than) group_size.
        s = inputs.shape  # [NHWC]  Input shape.
        y = tf.reshape(inputs, [group_size, -1, s[1], s[2], s[3]])  # [GMHWC] Split minibatch into M groups of size G.
        y = tf.cast(y, tf.float32)  # [GMHWC] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)  # [GMHWC] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)  # [MHWC]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)  # [MHWC]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[1, 2, 3], keepdims=True)  # [M111]  Take average over fmaps and pixels.
        y = tf.cast(y, inputs.dtype)  # [M111]  Cast back to original data type.
        y = tf.tile(y, [group_size, s[1], s[2], 1])  # [NHW1]  Replicate over group and pixels.
        return tf.concat([inputs, y], axis=-1)  # [NHWC]  Append as new fmap.

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2], input_shape[3] + 1


class Upscale2D(tf.keras.layers.Layer):
    """Nearest-neighbor upscaling layer. Non-trainable.
    TODO: Is it same as UpSampling2D(size, interpolation='nearest') ??? Try both and see behavior to decide?

    Pinched from Progressive GANs original repo:
    https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py"""

    def __init__(self, factor=2, name="Upscale2D", **kwargs):
        super().__init__(name=name, trainable=False, **kwargs)
        assert isinstance(factor, int) and factor >= 1
        self.factor = factor

    def call(self, x, **kwargs):
        if self.factor == 1:
            return x
        s = x.shape
        x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
        x = tf.tile(x, [1, 1, 1, self.factor, 1, self.factor])
        x = tf.reshape(x, [-1, s[1] * self.factor, s[2] * self.factor, s[3]])
        return x


class toRGBConv2D(tf.keras.layers.Conv2D):
    """Simply a convolutional layer with filters = number of colors, used to match dimensions at the end.
    See: Progressive GAN"""

    def __init__(self, number_of_colors, *args, **kwargs):
        super(toRGBConv2D, self).__init__(filters=number_of_colors, *args, **kwargs)


class Downscale2D(tf.keras.layers.Layer):
    """Box filter downscaling layer.
    Same goes as above (Upscale2D).
    """

    def __init__(self, factor=2, name="Downscale2D", **kwargs):
        super().__init__(name=name, trainable=False, **kwargs)
        assert isinstance(factor, int) and factor >= 1
        self.factor = factor

    def call(self, x, **kwargs):
        if self.factor == 1:
            return x
        ksize = [1, self.factor, self.factor, 1]
        return tf.nn.avg_pool(x, ksize=ksize, strides=ksize,
                              padding='VALID')  # NOTE: requires tf_config['graph_options.place_pruned_graph'] = True


class EqualizeLearningRate(tf.keras.layers.Wrapper):
    """
    ---Copied from https://github.com/henry32144/pggan-tensorflow/blob/master/modules.md---

    Reference from WeightNormalization implementation of TF Addons
    EqualizeLearningRate wrapper works for keras CNN and Dense (RNN not tested).
    ```python
      net = EqualizeLearningRate(
          tf.keras.submodels.Conv2D(2, 2, activation='relu'),
          input_shape=(32, 32, 3),
          data_init=True)(x)
      net = EqualizeLearningRate(
          tf.keras.submodels.Conv2D(16, 5, activation='relu'),
          data_init=True)(net)
      net = EqualizeLearningRate(
          tf.keras.submodels.Dense(120, activation='relu'),
          data_init=True)(net)
      net = EqualizeLearningRate(
          tf.keras.submodels.Dense(n_classes),
          data_init=True)(net)
    ```
    Arguments:
      layer: a layer instance.
    Raises:
      ValueError: If `Layer` does not contain a `kernel` of weights
    """

    def __init__(self, layer, **kwargs):
        super(EqualizeLearningRate, self).__init__(layer, **kwargs)
        self._track_trackable(layer, name='layer')
        self.is_rnn = isinstance(self.layer, tf.keras.layers.RNN)

    def build(self, input_shape=None):
        """Build `Layer`"""
        input_shape = tf.TensorShape(input_shape)
        self.input_spec = tf.keras.layers.InputSpec(
            shape=[None] + input_shape[1:])

        if not self.layer.built:
            self.layer.build(input_shape)

        kernel_layer = self.layer.cell if self.is_rnn else self.layer

        if not hasattr(kernel_layer, 'kernel'):
            raise ValueError('`EqualizeLearningRate` must wrap a layer that'
                             ' contains a `kernel` for weights')

        if self.is_rnn:
            kernel = kernel_layer.recurrent_kernel
        else:
            kernel = kernel_layer.kernel

        # He constant
        self.fan_in, self.fan_out = self._compute_fans(kernel.shape)
        self.he_constant = tf.Variable(1.0 / np.sqrt(self.fan_in), dtype=tf.float32, trainable=False)

        self.v = kernel
        self.built = True

    def call(self, inputs, training=True):
        """Call `Layer`"""

        with tf.name_scope('compute_weights'):
            # Multiply the kernel with the he constant.
            kernel = tf.identity(self.v * self.he_constant)

            if self.is_rnn:
                print(self.is_rnn)
                self.layer.cell.recurrent_kernel = kernel
                update_kernel = tf.identity(self.layer.cell.recurrent_kernel)
            else:
                self.layer.kernel = kernel
                update_kernel = tf.identity(self.layer.kernel)

            # Ensure we calculate result after updating kernel.
            with tf.control_dependencies([update_kernel]):
                outputs = self.layer(inputs)
                return outputs

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(
            self.layer.compute_output_shape(input_shape).as_list())

    def _compute_fans(self, shape, data_format='channels_last'):
        """
        From Official Keras implementation
        Computes the number of input and output units for a weight shape.
        # Arguments
            shape: Integer shape tuple.
            data_format: Image data format to use for convolution kernels.
                Note that all kernels in Keras are standardized on the
                `channels_last` ordering (even when inputs are set
                to `channels_first`).
        # Returns
            A tuple of scalars, `(fan_in, fan_out)`.
        # Raises
            ValueError: in case of invalid `data_format` argument.
        """
        if len(shape) == 2:
            fan_in = shape[0]
            fan_out = shape[1]
        elif len(shape) in {3, 4, 5}:
            # Assuming convolution kernels (1D, 2D or 3D).
            # TH kernel shape: (depth, input_depth, ...)
            # TF kernel shape: (..., input_depth, depth)
            if data_format == 'channels_first':
                receptive_field_size = np.prod(shape[2:])
                fan_in = shape[1] * receptive_field_size
                fan_out = shape[0] * receptive_field_size
            elif data_format == 'channels_last':
                receptive_field_size = np.prod(shape[:-2])
                fan_in = shape[-2] * receptive_field_size
                fan_out = shape[-1] * receptive_field_size
            else:
                raise ValueError('Invalid data_format: ' + data_format)
        else:
            # No copycats assumptions.
            fan_in = np.sqrt(np.prod(shape))
            fan_out = np.sqrt(np.prod(shape))
        return fan_in, fan_out


class ResNetBlock(tf.keras.layers.Wrapper):
    """Standard ResNet block.
            R(x) = Output — Input = H(x) — x
            => H(x) = R(x) + x
            The submodels in a traditional network are learning the true output H(x),
            whereas the submodels in a residual network are learning the residual R(x).
            Hence, the name: Residual Block.
    """

    def __init__(self, layer, shortcut_block=None, **kwargs):
        super(ResNetBlock, self).__init__(layer, **kwargs)
        self.add_layer = tf.keras.layers.Add()
        self.shortcut_block = shortcut_block

    def call(self, inputs, **kwargs):
        outputs = inputs
        outputs = self.layer(outputs)
        if self.shortcut_block is not None:
            inputs = self.shortcut_block(inputs)
        outputs = tf.keras.layers.LeakyReLU()(self.add_layer([outputs, inputs]))
        return outputs


class ResNetV2Block(tf.keras.layers.Wrapper):
    """S-o-d-a ResNetV2 block, with added pre-activation.
    """

    def __init__(self, layer, shortcut_block: Block = None, **kwargs):
        super(ResNetV2Block, self).__init__(layer, **kwargs)
        self.shortcut_block = shortcut_block
        self.norm_layer = tf.keras.layers.BatchNormalization()

    def call(self, inputs, **kwargs):
        outputs = inputs
        outputs = self.norm_layer(outputs)
        outputs = tf.keras.layers.ReLU()(outputs)
        outputs = self.layer(outputs)
        if self.shortcut_block is not None:
            inputs = self.shortcut_block(inputs)
        outputs = tf.keras.layers.Add()([outputs, inputs])
        outputs = tf.keras.layers.ReLU()(outputs)
        return outputs


class DuplicatingLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(DuplicatingLayer, self).__init__()
        self.trainable = False

    def call(self, inputs, **kwargs):
        return inputs, inputs


class ButtonLayer(tf.keras.layers.Layer):
    def __init__(self, button: tf.Variable, **kwargs):
        super(ButtonLayer, self).__init__(**kwargs)
        self.button = button

    def call(self, inputs, **kwargs):
        # TODO: Are we too eager?
        if self.button.numpy():
            return self.on(inputs)
        else:
            return self.off(inputs)

    def on(self, inputs):
        raise NotImplementedError

    def off(self, inputs):
        raise NotImplementedError


class OnOffButton(ButtonLayer):
    def __init__(self, layer, button: tf.Variable, **kwargs):
        super(OnOffButton, self).__init__(button, **kwargs)
        self.layer = layer

    def on(self, inputs):
        return self.layer(inputs)

    def off(self, inputs):
        return inputs


class Alpha:
    def __init__(self, total_steps=1000, **kwargs):
        self.total_steps = total_steps
        self.alpha = tf.Variable(initial_value=0, trainable=False, name="Alpha", dtype=tf.float32, **kwargs)
        self.increment_value = tf.constant(1. / total_steps, dtype=tf.float32)

    def increment(self):
        print("Incrementing Alpha")
        self.alpha.assign_add(self.increment_value)
        self.alpha.assign(tf.clip_by_value(self.alpha, clip_value_max=1.0, clip_value_min=0.0))

    def set_total_steps(self, total_steps):
        self.increment_value = tf.constant(1. / total_steps, dtype=tf.float32)


class AlphaMergerLayer(tf.keras.layers.Layer):
    def __init__(self, alpha: Alpha):
        super(AlphaMergerLayer, self).__init__()
        self.alpha = alpha
        self.add = tf.keras.layers.Add()
        self.multiply = tf.keras.layers.Lambda(lambda x: tf.math.multiply(x[0], x[1]))

    def call(self, inputs, **kwargs):
        input_main, input_residual = inputs
        input_main = self.multiply([self.alpha.alpha, input_main])
        input_residual = self.multiply([tf.Variable(1, dtype=tf.float32) - self.alpha.alpha, input_residual])
        tempres = self.add([input_main, input_residual])
        return tempres
