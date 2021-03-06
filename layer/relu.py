#!python3

import numpy as np
import matplotlib.pyplot as plt

class Relu:
    def __init__(self):
        pass
    def forward(self,x,train):
        return np.maximum(0,x)

    def backward(self,out,input,output):
        out[input<0] = 0
        return out

if __name__=="__main__":
    layer = Relu()
    x = np.arange(100) / 50 -1
    z = layer.forward(x,False)
    dz = np.gradient(z,x)
    out = layer.backward(z)
    plt.plot(x,z)
    plt.plot(x,dz)
    plt.plot(x,out)
    plt.show()
