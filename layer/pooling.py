import numpy as np

class MaxPooling:
    def __init__(self,shape):
        self.shape = shape
        pass

    def forward(self,x,train):
        row = self.shape[0]
        col = self.shape[1]
        out = x.copy().reshape(x.shape[0],x.shape[1]//row,row,x.shape[2]//col,col,x.shape[3])
        out = out.transpose(0,1,3,5,2,4)
        out = out.reshape(-1,row*col)
        maxpos = np.argmax(out,axis=1)
        out = out[np.arange(maxpos.size),maxpos]
        out = out.reshape(x.shape[0],x.shape[1]//row,x.shape[2]//col,x.shape[3])
        if train:
            self.maxpos = maxpos
        return out

    def backward(self,out,input,output):
        row = self.shape[0]
        col = self.shape[1]
        grad = np.zeros((out.size,row*col))
        grad[np.arange(self.maxpos.size),self.maxpos] = out.reshape(-1)
        grad = grad.reshape(out.shape[0],out.shape[1],out.shape[2],out.shape[3],row,col)
        grad = grad.transpose(0,1,4,2,5,3)
        grad = grad.reshape(input.shape)
        return grad

