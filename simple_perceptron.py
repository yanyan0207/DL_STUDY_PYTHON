from layer.softmax import Softmax
from layer.affin import Affine
from layer.relu import Relu

import numpy as np
import matplotlib.pyplot as plt


class simple_perceptron:
    def __init__(self,W,B):
        self.class_num = W.shape[1]
        self.layers = []
        self.layers.append(Affine(W=W,B=B)) 
        self.layers.append(Softmax()) 

    def predict(self,x,train=False):
        work = x
        for layer in self.layers:
            work = layer.foward(work,train)
        return work

    def loss(self,x,y,train=False,softmax=False):
        work = x
        for layer in self.layers:
            work = layer.foward(work,train)
        loss = np.sum(np.log(np.maximum(work[np.arange(len(y)),y],0.01))) * -1
        if softmax:
            return loss,work
        else:
            return loss

    def grad(self,x,y):
        h = 0.0001
        W = self.layers[0].W
        B = self.layers[0].B
        gradw = np.zeros(W.shape)
        gradb = np.zeros(B.shape)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                dw = np.zeros(W.shape)
                dw[i,j] = h
                loss_diff = simple_perceptron(W+dw,B).loss(x,y) - simple_perceptron(W-dw,B).loss(x,y)
                gradw[i,j] = loss_diff / h /2
        for i in range(B.shape[0]):
            db = np.zeros(B.shape)
            db[i] = h
            loss_diff = simple_perceptron(W,B+db).loss(x,y) - simple_perceptron(W,B-db).loss(x,y)
            gradb[i] = loss_diff / h /2
        return gradw,gradb

    def train(self,x,y):
        loss,softmax = self.loss(x,y,train=True,softmax=True)
        pred = np.argmax(softmax,axis=1)
        acc = np.sum(pred == y) / len(y)
        #print(self.softmax,y)
        print("loss",loss)
        print("acc",acc)
        work = np.eye(self.class_num)[y]
        for layer in reversed(self.layers):
            #print("backword",work)
            work = layer.backword(work)
        #print("backword",work)
        self.layers[0].update(-0.01)

if __name__=="__main__":
    x = np.random.rand(10000).reshape(-1,2) * 1 - 0.5
    y = np.array([0 if (a>b) else 1 for a,b in x])

    sp = simple_perceptron(W=np.array([[0.3,0.4],[0.6,0.7]]) - 0.5,B=np.zeros(2))
    #sp1 = simple_perceptron(W=sp.layers[0].W,B=[0.001,0])
    #sp2 = simple_perceptron(W=sp.layers[0].W,B=[-0.001,0])
    #print("lossdiff",sp1.loss(x,y,softmax=True),sp2.loss(x,y,softmax=True))
    print("W",sp.layers[0].W)
    print("B",sp.layers[0].B)
    for i in range(100):
        #print(sp.grad(x,y))
        sp.train(x,y)
        #print("dw",sp.layers[0].dw)
        #print("db",sp.layers[0].db)
        #print("W",sp.layers[0].W)
        #print("B",sp.layers[0].B)
    
    pred = np.argmax(sp.predict(x),axis=1)
    x0_x,x0_y = list(zip(*x[pred==0]))
    x1_x,x1_y = list(zip(*x[pred==1]))
    plt.scatter(x0_x,x0_y)
    plt.scatter(x1_x,x1_y)
    plt.show()