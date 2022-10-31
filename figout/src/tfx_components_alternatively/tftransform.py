
import tensorflow as tf


def transformed_name(key):
    return key + '_xf'


def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.
  Args:
    inputs: map from feature keys to raw not-yet-transformed features.
  Returns:
    Map from string feature key to transformed feature operations.
  """
    # TODO: I am pretty sure this is not how we access custom_config. The internet has failed me...
    outputs = {}
    image_xf = transformed_name('image')
    print("Image 0:", inputs['image'])

    def parse_image(img):
        img = tf.io.parse_tensor(tf.squeeze(img, axis=0), out_type=tf.float32)
        print("Image 1:", img)
        img = tf.reshape(img, (250, 250, 3))
        print("Image 2:", img)
        img = img / 255.
        print("Image 3:", img)
        return img

    outputs[image_xf] = tf.map_fn(parse_image, inputs['image'], dtype=tf.float32)
    return outputs
