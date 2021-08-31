import numpy as np

class MaxPooling:
    def __init__(self,shape):
        self.shape = shape
        pass

    def forward(self,x,train):
        row = self.shape[0]
        col = self.shape[1]
        out = x.reshape(x.shape[0],x.shape[1]//row,row,x.shape[2]//col,col,x.shape[3])
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

class MeanPooling:
    def __init__(self,shape):
        self.shape = shape
        pass

    def forward(self,x,train):
        row = self.shape[0]
        col = self.shape[1]
        out = x.reshape(x.shape[0],x.shape[1]//row,row,x.shape[2]//col,col,x.shape[3])
        out = out.transpose(0,1,3,5,2,4)
        out = out.reshape(x.shape[0],x.shape[1]//row,x.shape[2]//col,x.shape[3],row*col)
        out = np.average(out,axis=-1)
        return out

    def backward(self,out,input,output):
        row = self.shape[0]
        col = self.shape[1]
        out /= (row*col)
        print(out.shape)
        grad = out.transpose(0,3,1,2)
        grad = grad.reshape(grad.shape[0],grad.shape[1],-1,1)
        grad = grad.repeat(row*col,axis=3)
        grad = grad.reshape(grad.shape[0],grad.shape[1],output.shape[1],output.shape[2],row,col)
        grad = grad.transpose(0,2,4,3,5,1)
        grad = grad.reshape(input.shape)
        return grad

if __name__=="__main__":
    from PIL import Image
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train,axis=-1)
    pooling = MaxPooling((2,2))
    pooled = pooling.forward(x_train[:2],False)
    print(pooled.shape)
    pil_img = Image.fromarray(np.uint8(x_train[0,:,:,0]))
    pil_img2 = Image.fromarray(np.uint8(pooled[0,:,:,0]))
    pil_img.show()
    pil_img2.show()
    

