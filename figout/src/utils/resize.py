from collections import Iterable
from enum import Enum

import numpy as np
import tensorflow as tf


class ResizeMode(Enum):
    # ex. 80% of original size
    PERCENT = 0
    # ex. (1, 4) == 1/4 of original size
    RATIO = 1
    # ex. (4, 1) == 1/4 of original size
    INVERSE_RATIO = 2
    # ex. 0.25 of original size
    DECIMAL = 3
    # ex. (255, 255)
    FIXED = 4


class Resize:
    def __init__(self, size, mode: ResizeMode = ResizeMode.PERCENT):
        self.mode = mode
        self.size = size

    def __str__(self):
        if self.mode == ResizeMode.PERCENT:
            return f"{self.size}%"
        if self.mode == ResizeMode.RATIO:
            return f"{self.size[0]}/{self.size[1]}"
        if self.mode == ResizeMode.INVERSE_RATIO:
            return f"{self.size[1]}/{self.size[0]}"
        if self.mode == ResizeMode.DECIMAL:
            return f"{self.size}"
        if self.mode == ResizeMode.FIXED:
            return ", ".join(str(size) for size in self.size)

    def resize(self, orig_size):
        if isinstance(orig_size, tf.Tensor):
            pass
        if isinstance(orig_size, tuple) or isinstance(orig_size, list):
            orig_size = np.array(orig_size)
        elif not isinstance(orig_size, Iterable):
            orig_size = np.array([orig_size])
        for e, fn in zip(
            [ResizeMode.PERCENT, ResizeMode.RATIO, ResizeMode.INVERSE_RATIO, ResizeMode.FIXED, ResizeMode.DECIMAL],
            [self.percent_mode, self.ratio_mode, self.inverse_ratio_mode, self.fixed_mode, self.decimal_mode]
        ):
            if self.mode == e:
                res = fn(orig_size)
                res = [int(r) for r in res]
                return tuple(res)

    def percent_mode(self, orig_size):
        ratio = self.size / 100
        return orig_size * ratio

    def ratio_mode(self, orig_size):
        a, b = self.size
        return orig_size * (a / b)

    def inverse_ratio_mode(self, orig_size):
        b, a = self.size
        return orig_size * (a / b)

    def decimal_mode(self, orig_size):
        return orig_size * self.size

    def fixed_mode(self, orig_size):
        """TODO: If color channels is passed, that gives an error"""
        return self.size


class ImageResize(Resize):
    def resize_image(self, image):
        orig_size = tf.shape(image)[:-1]
        if len(orig_size) == 3:
            orig_size = orig_size[1:]
        new_size = self.resize(orig_size)
        image = tf.image.resize(images=image, size=new_size)
        return image
