#!python3

import numpy as np
import matplotlib.pyplot as plt

class Relu:
    def __init__(self):
        pass
    def foward(self,x,train):
        #print(x.shape)

        if train:
            self.mask = x < 0
        return np.maximum(0,x)
    def backword(self,out):
        dout = out.copy()
        dout[self.mask] = 0
        return dout

    
if __name__=="__main__":
    layer = Relu()
    x = np.arange(100) / 50 -1
    z = layer.foward(x)
    dz = np.gradient(z,x)
    out = layer.backword(z)
    plt.plot(x,z)
    plt.plot(x,dz)
    plt.plot(x,out)
    plt.show()
