import numpy as np

class Flatten:
    def forward(self,x,train):
        return x.reshape(x.shape[0],-1)
    def backward(seld,out,input,output):
        return out.reshape(input.shape)