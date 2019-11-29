"""[Inception test]
"""

import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from inception import *

# 环境设置
tf.random.set_seed(22)
np.random.seed(2019)
os.environ['IF_CPP_MIN_LOG_LEVEL'] = '2'

assert tf.__version__.startswith('2.')


if __name__ == "__main__":
    # 数据获取
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train.astype(
        np.float32)/255.0, x_test.astype(np.float32)/255.0
    x_train, x_test = np.expand_dims(
        x_train, axis=3), np.expand_dims(x_test, axis=3)

    db_train = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).batch(256)
    db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(256)

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    batch_size = 32
    epochs = 100
    model = Inception(2, 10)
    # deriver input shape for every layers
    model.build(input_shape=(None, 28, 28, 1))
    model.summary()

    optimizer = keras.optimizers.Adam(lr=1e-3)
    criteon = keras.losses.CategoricalCrossentropy(from_logits=True)

    acc_meter = keras.metrics.Accuracy()

    for epoch in range(100):
        for step, (x, y) in enumerate(db_train):
            with tf.GradientTape() as tape:
                logits = model(x)
                loss = criteon(tf.one_hot(y, depth=10), logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 10 == 0:
            print(epoch, step, 'loss:', loss.numpy())

        acc_meter.reset_states()
        for x, y in db_test:
            logits = model(x, training=False)
            pred = tf.argmax(logits, axis=1)
            acc_meter.update_state(y, pred)
        print(epoch, 'evalution acc:', acc_meter.result().numpy())
