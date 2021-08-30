#!python3

from os import error
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import copy

class CNN:
    def __init__(self,shape,channel_num,filter_num):
        self.W = (np.random.rand(shape[0]*shape[1]*channel_num,filter_num) - 0.5) * 0.01
        self.shape = shape
        self.channel_num = channel_num
        self.filter_num = filter_num

    def calc(self,x,row,col,shape,W):
        work = x[:,row:row+shape[0],col:col+shape[1],:]
        work = work.reshape(work.shape[0],-1)
        work = np.dot(work,W)
        return work

    def getWeights(self):
        return {
            "W" : self.W,
        }

    def setWeight(self,name,weight):
        if name == "W":
            self.W = weight
        else:
            raise error(name + " unkown weight name")

    def getWeightsGradient(self):
        return {
            "W" : self.dw,
        }

    def forward(self,x,train):
        out_rows = x.shape[1] - self.shape[0] + 1
        out_cols = x.shape[2] - self.shape[1] + 1
        out = [self.calc(x,row,col,self.shape,self.W) for row in range(out_rows) for col in range(out_cols)]
        out = np.array(out)
        out = out.transpose(1,0,2)
        out = out.reshape(x.shape[0],out_rows,out_cols,self.filter_num)
        return out

    def backword(self,out,input,output):
        # パディング
        pad_row = self.shape[0] - 1
        pad_col = self.shape[1] - 1

        #print("out",out)
        work = np.pad(out,[(0,0),(pad_row,pad_row),(pad_col,pad_col),(0,0)],"constant")
        #print("work padded",work)

        # 係数をひっくり返す
        #print("self.W.shape",self.W.shape)
        Wrev = self.W.reshape(-1,self.channel_num,self.filter_num)
        #print("Wrev",Wrev.shape,Wrev)
        Wrev = np.flipud(Wrev)
        #print("Wrev",Wrev.shape,Wrev)
        Wrev = Wrev.transpose(0,2,1)
        Wrev = Wrev.reshape(-1,self.channel_num)

        #
        #print("Wrev",Wrev.shape) 
        #print("work",work.shape)
        grad = [self.calc(work,row,col,self.shape,Wrev)
            for row in range(input.shape[1]) for col in range(input.shape[2])]
        grad = np.array(grad)
        #grad = grad.transpose(1,2,0)
        grad = grad.reshape(input.shape)

        self.dw = np.zeros((self.shape[0],self.shape[1],self.channel_num,self.filter_num))
        for filter in range(self.filter_num):
            worko = out[:,:,:,filter]
            for channel in range(self.channel_num):
                work_c = input[:,:,:,channel]
                for row in range(self.shape[0]):
                    for col in range(self.shape[1]):
                        worki = work_c[:,row:row+out.shape[1],col:col+out.shape[2]]
                        self.dw[row,col,channel,filter] = np.sum(worki*worko)
        
        self.dw = self.dw.reshape(-1,self.filter_num)
        return grad
    def update(self,alpha):
        self.W += alpha * self.dw


    def update(self,alpha):
        self.W += self.dw * alpha

if __name__=="__main__":
    layer = CNN((2,2),2,1)
    print(layer.W)
    layer.W = np.arange(8).T

    input = np.arange(16).reshape(2,2,2,2)
    output = layer.forward(input,True)
    print(output)
    print(layer.backword(output,input,output))
