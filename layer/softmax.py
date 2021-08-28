import numpy as np

class Softmax:
    def __init__(self):
        pass

    def foward(self,x,train):
        x = x.T
        x = x - np.max(x,axis=0)
        x = np.exp(x)
        softmax = x / np.sum(x,axis=0)
        softmax = softmax.T
        if train:
            self.softmax = softmax
        return softmax
    
    def backword(self,y):
        return self.softmax - y


