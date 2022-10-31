import numpy as np
# import cv2 as cv
import tensorflow as tf


def toGrayscale(images):
    w = np.array([[[0.07, 0.72, 0.21]]])
    return np.sum(images * w, axis=-1)

def getSobelTF(images):
    images = tf.image.sobel_edges(images)
    images = images * images
    images = tf.math.reduce_sum(images, axis=-1)
    return tf.sqrt(images)

def getLaplacianTF(images):
    images = getSobelTF(images)
    return getSobelTF(images)

def tensor_to_image(tensor):
    image = (tensor + 1) / 2
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    return image

def image_to_tensor(image):
    tensor = tf.image.convert_image_dtype(image, dtype=tf.float32)
    tensor = (tensor * 2) - 1
    return tensor

def image_file_to_tensor(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_image(image)
    tensor = image_to_tensor(image)
    return tensor
