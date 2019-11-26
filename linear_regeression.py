import tensorflow as tf
import numpy as np
from tensorflow import  keras
import os

class Regressor(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.w = self.add_variable('w',[13,1])
        self.b = self.add_variable('b',[1])
        print(self.w.shape,self.b.shape)
    
    def call(self,x):
        x = tf.matmul(x,self.w) + self.b
        return x

def main():
    tf.random.set_seed(22)
    np.random.seed(22)
    (x_train,y_train),(x_val,y_val) = keras.datasets.boston_housing.load_data()
    print(x_train.shape,y_train.shape)
    print(x_val.shape,y_val.shape)

    db_train = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(64)
    db_val = tf.data.Dataset.from_tensor_slices((x_val,y_val)).batch(102)

    model = Regressor()
    criteon = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(lr=1e-2)

    for epoch in range(200):
        for step,(x,y) in enumerate(db_train):
            with tf.GradientTape() as tape:
                logits = model(x)
                logits = tf.squeeze(logits,axis=1)
                loss = criteon(y,logits)
            grads = tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(grads,model.trainable_variables))
        
        print(epoch,'trn_loss:',loss.numpy())

        if epoch %10 == 0:
            for x,y in db_val:
                logits = model(x)
                logits = tf.squeeze(logits,axis=1)
                loss = criteon(y,logits)
                print(epoch,'val_loss:',loss.numpy())
    
if __name__ == "__main__":
    main()



