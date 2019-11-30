import os
import tensorflow as tf
import numpy as np
from tensorflow import keras


class VAE(keras.Model):
    """ Auto-Encoding Variational Bayes
        https://arxiv.org/pdf/1312.6114.pdf
    Arguments:
        keras {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    def __init__(self,h_dim=512,z_dim=20,image_size=28*28):
        super().__init__()
        self.fc1 = keras.layers.Dense(h_dim)
        self.fc2 = keras.layers.Dense(z_dim)
        self.fc3 = keras.layers.Dense(z_dim)

        self.fc4 = keras.layers.Dense(h_dim)
        self.fc5 = keras.layers.Dense(image_size)

    def encode(self,x):
        h = tf.nn.relu(self.fc1(x))
        return self.fc2(h),self.fc3(h)

    def reparameterize(self,mu,log_var):
        std = tf.exp(log_var*0.5)
        eps = tf.random.normal(std.shape)
        return mu + eps * std
    
    def decode_logits(self,z):
        h = tf.nn.relu(self.fc4(z))
        return self.fc5(h)
    
    def decode(self,z):
        return tf.nn.sigmoid(self.decode_logits(z))
    
    def call(self,inputs,training=None,mask=None):
        mu,log_var = self.encode(inputs)
        z = self.reparameterize(mu,log_var)

        x_reconstructed_logits = self.decode_logits(z)

        return x_reconstructed_logits,mu,log_var

if __name__ == "__main__":
    model = VAE()
    model.build(input_shape=(4,28*28))
    model.summary()


