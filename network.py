import numpy as np
import matplotlib.pyplot as plt
from numpy.testing._private.utils import print_assert_equal
from utils.measure_time import *
from collections import OrderedDict

class Network:
    def __init__(self,class_num):
        self.layers = OrderedDict()
        self.historys = []
        self.class_num = class_num
    def addLayer(self,layer):
        self.layers[type(layer).__name__ + "_layer_" + str(len(self.layers) + 1)] = layer

    def predict(self,x):
        work = x
        for key,layer in self.layers.items():
            work = layer.forward(work,False)
        return work

    def loss(self,x,y,train=False,softmax=False):
        self.data_list = [x]
        work = x
        for name,layer in self.layers.items():
            startTime("train loss:" + name)
            work = layer.forward(work.copy(),train)
            self.data_list.append(work)
            endTime("train loss:" + name)

        startTime("logloss:")
        loss = np.average(np.log(np.maximum(work[np.arange(len(y)),y],1e-7))) * -1
        endTime("logloss:")

        if softmax:
            return loss,work
        else:
            return loss

    def train(self,x,y,update_params=True):
        loss,softmax = self.loss(x,y,train=True,softmax=True)
        pred = np.argmax(softmax,axis=1)
        acc = np.sum(pred == y) / len(y)
        #print(self.softmax,y)
        print("\racc",acc,"loss",loss,end="")
        self.historys.append((loss,acc))
        work = np.eye(self.class_num)[y]

        for i,(name,layer) in enumerate(reversed(self.layers.items())):
            startTime("train backward:" + name)
            input = self.data_list[len(self.layers) - i - 1]
            output = self.data_list[len(self.layers) - i]

            #print("backward",work)
            work = layer.backward(work.copy(),input.copy(),output.copy())
            endTime("train backward:" + name)
        #print("backward",work)

        if update_params:
            for key, layer in self.layers.items():
                if getattr(layer,"update", None):
                    layer.update(-0.1)

    def getWeights(self):
        weighs = OrderedDict()
        for name,layer in self.layers.items():
            if getattr(layer,"getWeights",None):
                weighs[name] = layer.getWeights()
        return weighs

    def setWeights(self,weights):
        for name,weight in weights.items():
            self.layer[name].setWeights(weight)

    def getWeightsGradient(self):
        self.weighs_grad = OrderedDict()
        for name,layer in self.layers.items():
            if getattr(layer,"getWeightsGradient",None):
                self.weighs_grad[name] = layer.getWeightsGradient()
        return self.weighs_grad

    def calcGradient(self,x,y):
        h = 1e-5
        weights_list = self.getWeights()
        dw = OrderedDict()
        for name, weights in weights_list.items():
            dw[name] = {}
            for wname, W in weights.items():                
                grads = np.zeros(W.size)
                for idx in range(W.size):
                    Wtest = W.copy().reshape(-1)
                    Wtest[idx] += h
                    Wtest = Wtest.reshape(W.shape)
                    self.layers[name].setWeight(wname,Wtest)
                    lossh = self.loss(x,y)
                    Wtest = W.copy().reshape(-1)
                    Wtest[idx] -= h
                    Wtest = Wtest.reshape(W.shape)
                    self.layers[name].setWeight(wname,Wtest)
                    lossl = self.loss(x,y)
                    grads[idx] = (lossh - lossl) / h / 2
                    self.layers[name].setWeight(wname,W)
                dw[name][wname] = grads.reshape(W.shape)
        return dw

if __name__=="__main__":
    from layer.affin import Affine
    from layer.relu import Relu
    from layer.softmax import Softmax

    x = np.random.rand(10000).reshape(-1,2) * 2 - 1
    y = np.array([0 if (a * b >= 0 and a*a +b * b<1) else 1 for a,b in x])

    sp = Network(2)
    sp.addLayer(Affine(in_num=2,out_num = 50))
    sp.addLayer(Relu())
    sp.addLayer(Affine(in_num=50,out_num = 10))
    sp.addLayer(Relu())
    sp.addLayer(Affine(in_num=10,out_num = 2))
    sp.addLayer(Softmax())
    CALC_DW=False
    for i in range(1000):
        if CALC_DW:
            calc_dw = sp.calcGradient(x,y)
        sp.train(x,y)
        if CALC_DW:
            dw = sp.getWeightsGradient()
            for name,grads_layer in dw.items():
                for wname,grads in grads_layer.items():
                    grads = grads.reshape(-1)
                    calc_grads = calc_dw[name][wname].reshape(-1)
                    for idx,(g,cg) in enumerate(zip(grads,calc_grads)):
                        if g == 0 and cg == 0:
                            pass
                        elif g== 0 or cg == 0:
                            print("warn " ,name,wname,idx,"include zero")
                        elif not 1.1 > g / cg > 0.9: 
                            print("warn " ,name,wname,idx,"differnt",grads,calc_grads)
                            continue
    print("W",sp.getWeights())
    pred = np.argmax(sp.predict(x),axis=1)
    for (p,t) in [(p,t) for p in [0,1] for t in [0,1]]:
        work = x[(y==t) & (pred==p)]
        if len(work) > 0:
            x1,x2 = list(zip(*work))
            plt.scatter(x1,x2)
    printTime()
    plt.figure()
    losses,accs = list(zip(*sp.historys))
    plt.plot(losses)
    plt.twinx()
    plt.plot(accs,c="r")
    plt.show()