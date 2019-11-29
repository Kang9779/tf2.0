"""ResNet Model
"""
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras

# 3x3 convolution


def conv3x3(channels, stride=1, kernel=(3, 3)):
    return keras.layers.Conv2D(channels, kernel, strides=stride, padding='same',
                               use_bias=False,
                               kernel_initializer=tf.random_normal_initializer())


class ResnetBlock(keras.Model):

    def __init__(self, channels, strides=1, residual_path=False):
        super().__init__()

        self.channels = channels
        self.strides = strides
        self.residual_path = residual_path

        self.conv1 = conv3x3(channels, strides)
        self.bn1 = keras.layers.BatchNormalization()
        self.conv2 = conv3x3(channels)
        self.bn2 = keras.layers.BatchNormalization()

        if residual_path:
            self.down_conv = conv3x3(channels, strides, kernel=(1, 1))
            self.down_bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None):
        residual = inputs

        x = self.bn1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv1(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)

        # this module can be added into self.
        # however, module in for can not be added.
        if self.residual_path:
            residual = self.down_bn(inputs, training=training)
            residual = tf.nn.relu(residual)
            residual = self.down_conv(residual)

        x = x + residual
        return x


class ResNet(keras.Model):

    def __init__(self, block_list, num_classes, initial_filters=16, **kwargs):
        super().__init__(**kwargs)

        self.num_blocks = len(block_list)
        self.block_list = block_list

        self.in_channels = initial_filters
        self.out_channels = initial_filters
        self.conv_initial = conv3x3(self.out_channels)

        self.blocks = keras.models.Sequential(name='dynamic-blocks')

        # build all the blocks
        for block_id in range(len(block_list)):
            for layer_id in range(block_list[block_id]):

                if block_id != 0 and layer_id == 0:
                    block = ResnetBlock(self.out_channels,
                                        strides=2, residual_path=True)
                else:
                    if self.in_channels != self.out_channels:
                        residual_path = True
                    else:
                        residual_path = False
                    block = ResnetBlock(self.out_channels,
                                        residual_path=residual_path)

                self.in_channels = self.out_channels

                self.blocks.add(block)

            self.out_channels *= 2

        self.final_bn = keras.layers.BatchNormalization()
        self.avg_pool = keras.layers.GlobalAveragePooling2D()
        self.fc = keras.layers.Dense(num_classes)

    def call(self, inputs, training=None):
        out = self.conv_initial(inputs)
        out = self.blocks(out, training=training)
        out = self.final_bn(out, training=training)
        out = tf.nn.relu(out)
        out = self.avg_pool(out)
        out = self.fc(out)
        return out


if __name__ == "__main__":
    model = ResNet([2, 2, 2], 10)
    model.build(input_shape=(None, 28, 28, 1))
    model.summary()
