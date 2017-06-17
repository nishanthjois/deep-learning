import os
import os.path as path

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

MODEL_NAME='mnist_convnet'
EPOCHS = 10
BATCH_SIZE=2

def load_data():
    (x_train,y_train),(x_test,y_test) = mnist.load_data()
    x_train=x_train.reshape(x_train.shape[0],28,28,1)
    
test=load_data()
    