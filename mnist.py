from network import Network
from keras.datasets import mnist
from layer.affin import Affine
from layer.softmax import Softmax
from layer.relu import Relu
import numpy as np
from utils.measure_time import *
import sys

if __name__=="__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print (x_train.shape,y_train.shape)
    x_train = x_train.reshape(x_train.shape[0],-1)
    x_test = x_test.reshape(x_test.shape[0],-1)

    x_train = x_train / 256
    x_test = x_test / 256


    network = Network(class_num=10)
    network.addLayer(Affine(x_train.shape[1],128))
    network.addLayer(Relu())
    network.addLayer(Affine(128,10))
    network.addLayer(Softmax())

    for i in range(100):
        startTime("train")
        network.train(x_train,y_train)
        endTime("train")
        startTime("test")
        pred = np.argmax(network.predict(x_test),axis=1)
        endTime("test")
        print(i,np.sum(pred==y_test)/len(y_test))

    printTime()