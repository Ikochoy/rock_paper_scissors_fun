import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import utils
import matplotlib.pyplot as plt
import cv2
import os

class EpochCallBack(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.99:
            print("reached 99% accuracy")
            self.model.stop_training = True


class DataSetBuilder():
    def __init__(self):
        self.dataset = {}
        for directory in os.listdir('training_images'):
            if directory.isdigit():
                self.dataset[directory] = []
                path = os.path.join('training_images', directory)
                for item in os.listdir(path):
                    if not item.startswith('.'):
                        img = cv2.imread(os.path.join(path, item))
                        img = cv2.resize(img, (300, 300))
                        self.dataset[directory].append(img)

                                 
    def get_dataset(self):
        labels = []
        images = []
        for label in self.dataset:
            for img in self.dataset[label]:
                images.append(img)
                labels.append(label)
        return images, labels
        
    
        
class NeuralNetwork():
    def __init__(self):
        self.model = keras.Sequential([
                keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu, input_shape=(300, 300, 3)),
                keras.layers.MaxPooling2D(2, 2),
                keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu),
                keras.layers.MaxPooling2D(2, 2),
                keras.layers.Conv2D(128, (3,3), activation=tf.nn.relu),
                keras.layers.MaxPooling2D(2, 2),
                keras.layers.Conv2D(128, (3,3), activation=tf.nn.relu),
                keras.layers.MaxPooling2D(2, 2),     
                keras.layers.Flatten(),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(512, activation=tf.nn.relu),
                keras.layers.Dense(4, activation=tf.nn.softmax)
              ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    def train_neural_network(self, img_train, label_train):
        callback = EpochCallBack()
        self.model.fit(x=img_train, y=label_train , epochs=20, callbacks=[callback])
       


if __name__ == '__main__': 
    db = DataSetBuilder()
    img_ds, lb_ds = db.get_dataset()
    # one hot encode the labels
    lb_ds = utils.to_categorical(lb_ds, 4)
    nn = NeuralNetwork()
    nn.train_neural_network(np.array(img_ds), np.array(lb_ds))
    nn.model.save('saved_model/rps_model.h5') 
    
    

