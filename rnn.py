import  os
import tensorflow as tf
from tensorflow import keras
import numpy as np


top_words = 10000
max_review_length = 80

class RNN(keras.Model):
    """[summary]
    
    Arguments:
        keras {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    def __init__(self,units,num_classes,num_layers=None):
        super().__init__()
        """[summary]
        """
        self.rnn = keras.layers.LSTM(units,return_sequences=True)
        self.rnn2 = keras.layers.LSTM(units)

        self.embedding = keras.layers.Embedding(top_words,100,input_length=max_review_length)
        self.fc = keras.layers.Dense(1)
    
    def call(self,inputs,training=None,mask=None):
        """[summary]
        
        Arguments:
            inputs {[type]} -- [description]
        
        Keyword Arguments:
            training {[type]} -- [description] (default: {None})
            mask {[type]} -- [description] (default: {None})
        
        Returns:
            [type] -- [description]
        """
        x = self.embedding(inputs)
        out = self.rnn(x)
        out = self.rnn2(out)
        out = self.fc(out)
        print(out.shape)
        return out

if __name__ == "__main__":

    (x_train,y_train),(x_test,y_test) = keras.datasets.imdb.load_data(num_words=top_words)
    
    x_train = keras.preprocessing.sequence.pad_sequences(x_train,maxlen=max_review_length)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test,maxlen=max_review_length)

    print(x_train.shape)
    print(x_test.shape)

    units = 64
    num_classes = 2
    batch_size = 32
    epochs = 20

    model = RNN(units,num_classes,num_layers=2)

    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss = keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.fit(
        x_train,y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test,y_test),
        verbose=1
    )
    scores = model.evaluate(x_test,y_test,batch_size,verbose=1)
    print("Test loss and Accuracy:",scores)
