#!python3

import numpy as np
import matplotlib.pyplot as plt

class Affine:
    def __init__(self,in_num=None,out_num=None,W=None,B=None):
        self.W = np.random.rand(in_num*out_num).reshape(in_num,out_num) - 0.5 if W is None else np.array(W)
        self.B = np.random.rand((out_num)) - 0.5 if B is None else np.array(B) 

    def foward(self,x,train):
        return np.dot(x,self.W) + self.B

    def backword(self,out,input,output):
        #print("Affine1 backword out",out)
        # biasの勾配は下層の勾配そのまま
        self.db = np.sum(out,axis=0)
        # Wの勾配は。。。
        self.dw = np.dot(input.T,out)

        grad = np.dot(out,self.W.T)
        return grad

    def update(self,alpha):
        self.B += self.db * alpha
        self.W += self.dw * alpha
    
if __name__=="__main__":
    layer = Affine(W=[[1,2],[3,4],[5,6]],B=[1,2])
    x = [[1,2,3]]
    output = layer.foward(x,True)
    print(output)
    dout = layer.backword(output)
    print(dout)
