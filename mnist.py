from network import Network
from keras.datasets import mnist
from layer.affin import Affine
from layer.softmax import Softmax
from layer.relu import Relu
from layer.cnn import CNN
from layer.pooling import *
from layer.shape import *
import numpy as np
from utils.measure_time import *
import sys

def calc_dw_test(x,y):
    print("y",y)
    calc_dw = network.calcGradient(x,y)
    network.train(x,y,update_params=False)
    dw = network.getWeightsGradient()
    for name,grads_layer in dw.items():
        for wname,grads in grads_layer.items():
            grads = grads.reshape(-1)
            calc_grads = calc_dw[name][wname].reshape(-1)
            for idx,(g,cg) in enumerate(zip(grads,calc_grads)):
                if np.abs(g) < 1e-9 and np.abs(cg) < 1e-9:
                    pass
                elif g== 0 or cg == 0:
                    print("warn " ,name,wname,idx,"include zero",grads[idx],calc_grads[idx])
                elif not 1.1 > g / cg > 0.9: 
                    print("warn " ,name,wname,idx,"differnt",grads[idx],calc_grads[idx])
                    continue

if __name__=="__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print (x_train.shape,y_train.shape)
    #x_train = x_train[:1]
    #y_train = y_train[:1]
    x_train = x_train / 256 - 0.5
    x_test = x_test / 256 - 0.5


    network = Network(class_num=10)
    if True:
        network.addLayer(CNN(shape=(3,3),channel_num = 1,filter_num=2))
        network.addLayer(MaxPooling(shape=(2,2)))
        network.addLayer(Relu())
        network.addLayer(Flatten())
        network.addLayer(Affine(169*2,10))
        network.addLayer(Softmax())

        x_train = np.expand_dims(x_train,-1)
        x_test = np.expand_dims(x_test,-1)
    else:
        x_train = x_train.reshape(-1,784)
        x_test = x_test.reshape(-1,784)
        network.addLayer(Affine(784,128))
        network.addLayer(Relu())
        network.addLayer(Affine(128,10))
        network.addLayer(Softmax())

    CALC_DW_SIZE=1
    MINI_BATCH_SIZE=1
    EPOCH=1

    lastacc = 0
    for i in range(EPOCH):
        if CALC_DW_SIZE > 0:
            startTime("calc dw")
            calc_dw_test(x_train[:CALC_DW_SIZE],y_train[:CALC_DW_SIZE])
            endTime("calc dw")
        for j in range(len(x_train) // MINI_BATCH_SIZE):
            x_train_batch = x_train[MINI_BATCH_SIZE*j:MINI_BATCH_SIZE*(j+1)]
            y_train_batch = y_train[MINI_BATCH_SIZE*j:MINI_BATCH_SIZE*(j+1)]
            startTime("train")
            network.train(x_train_batch,y_train_batch)
            endTime("train")
            
            if j % 100 == 0:
                startTime("test")
                print(j,"lastacc",lastacc)
                preds = np.zeros(len(x_test))
                for k in range(len(x_test) // MINI_BATCH_SIZE):   
                    pred = np.argmax(network.predict(x_test[k*MINI_BATCH_SIZE:(k+1)*MINI_BATCH_SIZE]),axis=1)
                    preds[k*MINI_BATCH_SIZE:(k+1)*MINI_BATCH_SIZE] = pred
                print(j,np.sum(preds==y_test)/len(y_test))
                lastacc = np.sum(preds==y_test)/len(y_test)
                endTime("test")

    printTime()