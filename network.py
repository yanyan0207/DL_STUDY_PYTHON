import numpy as np
import matplotlib.pyplot as plt

class Network:
    def __init__(self,class_num):
        self.layers = []
        self.historys = []
        self.class_num = class_num
    def addLayer(self,layer):
        self.layers.append(layer)

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

    def train(self,x,y):
        loss,softmax = self.loss(x,y,train=True,softmax=True)
        pred = np.argmax(softmax,axis=1)
        acc = np.sum(pred == y) / len(y)
        #print(self.softmax,y)
        print("loss",loss)
        print("acc",acc)
        self.historys.append((loss,acc))
        work = np.eye(self.class_num)[y]
        for layer in reversed(self.layers):
            #print("backword",work)
            work = layer.backword(work)
        #print("backword",work)

        for layer in self.layers:
            if getattr(layer,"update", None):
                layer.update(-0.0001)

if __name__=="__main__":
    from layer.affin import Affine
    from layer.relu import Relu
    from layer.softmax import Softmax

    x = np.random.rand(10000).reshape(-1,2) * 2 - 1
    y = np.array([0 if (a > 0 and b > 0 and a*a +b * b<1) else 1 for a,b in x])

    sp = Network(2)
    sp.addLayer(Affine(in_num=2,out_num = 10))
    sp.addLayer(Relu())
    sp.addLayer(Affine(in_num=10,out_num = 5))
    sp.addLayer(Relu())
    sp.addLayer(Affine(in_num=5,out_num = 2))
    sp.addLayer(Softmax())

    for i in range(1000):
        #print(sp.grad(x,y))
        sp.train(x,y)
        #print("dw",sp.layers[0].dw)
        #print("db",sp.layers[0].db)
        #print("W",sp.layers[0].W)
        #print("B",sp.layers[0].B)
    
    pred = np.argmax(sp.predict(x),axis=1)
    for (p,t) in [(p,t) for p in [0,1] for t in [0,1]]:
        work = x[(y==t) & (pred==p)]
        if len(work) > 0:
            x1,x2 = list(zip(*work))
            plt.scatter(x1,x2)
    plt.figure()
    losses,accs = list(zip(*sp.historys))
    plt.plot(losses)
    plt.twinx()
    plt.plot(accs,c="r")
    plt.show()