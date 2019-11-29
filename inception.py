"""Inception Network
"""
import tensorflow as tf
import numpy as np
from tensorflow import keras


class ConvBNRelu(keras.Model):
    def __init__(self, ch, kernel_size=3, strides=1, padding='same'):
        super().__init__()
        self.model = keras.models.Sequential([
            keras.layers.Conv2D(
                ch, kernel_size, strides=strides, padding=padding),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU()
        ])

    def call(self, x, training=None):
        x = self.model(x, training=training)
        return x


class InceptionBlock(keras.Model):
    def __init__(self, ch, strides=1):
        """[summary]

        Arguments:
            keras {[type]} -- [description]
            ch {[type]} -- [description]

        Keyword Arguments:
            strides {int} -- [description] (default: {1})
        """
        super().__init__()
        self.ch = ch
        self.strides = strides
        self.conv1 = ConvBNRelu(ch, strides=strides)
        self.conv2 = ConvBNRelu(ch, kernel_size=3, strides=strides)
        self.conv3_1 = ConvBNRelu(ch, kernel_size=3, strides=strides)
        self.conv3_2 = ConvBNRelu(ch, kernel_size=3, strides=1)

        self.pool = keras.layers.MaxPooling2D(3, strides=1, padding='same')
        self.pool_conv = ConvBNRelu(ch, strides=strides)

    def call(self, x, training=None):
        """[summary]

        Arguments:
            x {[type]} -- [description]

        Keyword Arguments:
            training {[type]} -- [description] (default: {None})

        Returns:
            [type] -- [description]
        """
        x1 = self.conv1(x, training=training)
        x2 = self.conv2(x, training=training)

        x3_1 = self.conv3_1(x, training=training)
        x3_2 = self.conv3_2(x3_1, training=training)

        x4 = self.pool(x)
        x4 = self.pool_conv(x4, training=training)

        x = tf.concat([x1, x2, x3_2, x4], axis=3)
        return x


class Inception(keras.Model):
    def __init__(self, num_layers, num_classes, init_ch=16, **kwargs):
        """[summary]

        Arguments:
            keras {[type]} -- [description]
            num_layers {[type]} -- [description]
            num_classes {[type]} -- [description]

        Keyword Arguments:
            init_ch {int} -- [description] (default: {16})
        """
        super().__init__(**kwargs)
        self.in_channels = init_ch
        self.out_channels = init_ch
        self.num_layers = num_layers
        self.init_ch = init_ch

        self.conv1 = ConvBNRelu(init_ch)

        self.blocks = keras.models.Sequential(name='dynamic_blocks')

        for block_id in range(num_layers):
            for layer_id in range(2):
                if layer_id == 0:
                    block = InceptionBlock(self.out_channels, strides=2)
                else:
                    block = InceptionBlock(self.out_channels, strides=1)
                self.blocks.add(block)
            self.out_channels *= 2
        self.avg_pool = keras.layers.GlobalAveragePooling2D()
        self.fc = keras.layers.Dense(num_classes)

    def call(self, x, training=None):
        out = self.conv1(x, training=training)
        out = self.blocks(out, training=training)
        out = self.avg_pool(out)
        out = self.fc(out)
        return out
