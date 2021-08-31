#!python3

from os import error
from utils.measure_time import *
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import copy

class CNN:
    def __init__(self,shape,channel_num,filter_num):
        self.W = (np.random.rand(shape[0]*shape[1]*channel_num,filter_num) - 0.5)
        self.shape = shape
        self.channel_num = channel_num
        self.filter_num = filter_num

    def calc(self,x,row,col,shape,W):
        work = x[:,row:row+shape[0],col:col+shape[1],:]
        work = work.reshape(work.shape[0],-1)
        work = np.dot(work,W)
        return work

    def im2col(self,x,shape):
        startTime("im2col")
        out = np.asarray([x[:,row:x.shape[1]-shape[0]+row+1,col:x.shape[2]-shape[1] + col+1,:]
             for row in range(shape[0]) for col in range(shape[1])])
        out = out.transpose(1,2,3,0,4)
        out = out.reshape(out.shape[0],out.shape[1],out.shape[2],-1)
        endTime("im2col")
        return out

    def conv(self,x,shape,W, use_im2col=False):
        startTime("conv")
        if False:
            out = self.im2col(x,shape)
            out = np.dot(out,W)
        elif True:
            out_rows = x.shape[1] - shape[0] + 1
            out_cols = x.shape[2] - shape[1] + 1
            out = [self.calc(x,row,col,shape,W) for row in range(out_rows) for col in range(out_cols)]
            out = np.array(out)
            out = out.transpose(1,0,2)
            out = out.reshape(x.shape[0],out_rows,out_cols,W.shape[1])
        else:
            out = np.zeros((x.shape[0],x.shape[1] - shape[0] + 1,x.shape[2] - shape[1] + 1,W.shape[1]))
            W = W.reshape(shape[0],shape[1],-1,W.shape[1])
            for filter in range(W.shape[3]):
                Wfilter = W[:,:,:,filter]
                for col in range(x.shape[2] - shape[1] + 1):
                    xcol = x[:,:,col:col+shape[1],:]
                    for row in range(x.shape[1] - shape[0] + 1):
                        out[:,row,col,filter]  = np.sum(xcol[:,row:row+shape[0],:,:] * Wfilter,axis=(1,2,3))

        endTime("conv")
        return out

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
        return self.conv(x,self.shape,self.W)

    def backward(self,out,input,output):
        startTime("cnn backward grad")
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
        grad = self.conv(work,self.shape,Wrev)

        endTime("cnn backward grad")
        startTime("cnn backward dw")
        if True:
            self.dw = np.zeros((self.shape[0],self.shape[1],self.channel_num,self.filter_num))
            for filter in range(self.filter_num):
                worko = out[:,:,:,filter]
                for channel in range(self.channel_num):
                    work_c = input[:,:,:,channel]
                    for row in range(self.shape[0]):
                        for col in range(self.shape[1]):
                            worki = work_c[:,row:row+out.shape[1],col:col+out.shape[2]]
                            self.dw[row,col,channel,filter] = np.sum(worki*worko)
        else:
            # INPUT Channel x row x col x MiniBatch
            work_i = input.transpose(3,1,2,0)
            # OUTPUT row x col x MiniBatch x filter_num
            work_o = out.transpose(1,2,0,3)
            self.dw = self.conv(work_i,(work_o.shape[0],work_o.shape[1]),work_o.reshape(-1,work_o.shape[3]),True)
            self.dw = self.dw.transpose(1,2,0,3)

        self.dw = self.dw.reshape(-1,self.filter_num)
        endTime("cnn backward dw")
        return grad

    def update(self,alpha):
        self.W += alpha * self.dw

    def update(self,alpha):
        self.W += self.dw * alpha

if __name__=="__main__":
    layer = CNN((2,2),1,1)
    print(layer.W)
    W = np.arange(1,5).reshape(4,1)
    layer.W = W.copy()

    input = np.arange(9).reshape(1,3,3,1)
    print("W",layer.W)
    print("input",input)

    output = layer.forward(input,True)
    print(output)
    out = np.arange(1,5).reshape(1,2,2,1)
    print(layer.backward(out,input,output))

    base = np.sum(out * output)
    grad = np.zeros(input.size)
    for i in range(input.size):
        layer.W = W.copy()
        input_copy = input.copy()
        input_copy = input_copy.reshape(-1)
        input_copy[i] += 1
        input_copy = input_copy.reshape(input.shape)
        output = layer.forward(input_copy,True)
        grad[i] = np.sum(output * out) - base

    print("calc grad",grad)