import sys
import numpy as np
import canton as ct
from canton import *
import tensorflow as tf
from scipy.misc import imread
import math

def Mnist():
    from keras.datasets import fashion_mnist
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()#32*32的数据集
    X_train = X_train[:,:,:, np.newaxis]
    X_test=X_test[:,:,:, np.newaxis]
    print('X_train shape:', X_train.shape)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255

    return X_train,X_test

def ComCNN():
    c=Can()
    def conv(nip,nop,flag=True):
        c.add(Conv2D(nip,nop,k=3,usebias=True))
        if flag:
            # c.add(BatchNorm(nop))
            c.add(Act('elu'))
    c.add(Lambda(lambda x:x-0.5))
    conv(1,32)
    conv(32,64)
    conv(64,128)
    conv(128,256)
    conv(256, 128)
    conv(128,64)
    conv(64, 64)
    conv(64,32)
    conv(32, 4,flag=False)
    c.chain()
    return c

def ResCNN():
    c=Can()
    def conv(nip,nop,flag=True):
        c.add(Conv2D(nip,nop,k=3,usebias=True))
        if flag:
            # c.add(BatchNorm(nop))
            c.add(Act('elu'))
    conv(4,32)
    conv(32,64)
    conv(64,128)
    conv(128, 256)
    conv(256, 128)
    conv(128,64)
    conv(64,64)
    conv(64,32)
    conv(32, 1, flag=False)
    c.add(Act('sigmoid'))
    c.chain()
    return c

def get_trainer():
    x = ph([None,None,1])
    x = tf.clip_by_value(x,clip_value_max=1.,clip_value_min=0.)
    code_noise = tf.Variable(1.0)
    linear_code = com(x)

    # add gaussian before sigmoid to encourage binary code
    noisy_code = linear_code - \
        tf.random_normal(stddev=code_noise,shape=tf.shape(linear_code))
    binary_code = Act('sigmoid')(noisy_code)
    y = res(binary_code)
    set_training_state(False)
    quantization_threshold = tf.Variable(0.5)
    binary_code_test = tf.cast(binary_code>quantization_threshold,tf.float32)
    y_test = res(binary_code_test)

    def test(batch,quanth):
        sess = ct.get_session()
        res = sess.run([binary_code_test,y_test,binary_code,y,x],feed_dict={
            x:batch,
            quantization_threshold:quanth,
        })
        return res
    return test

def change(x):
    x *=255
    x=x.astype('uint8')
    img=x[0]
    return img

def psnr(im1,im2):
    mse = np.mean( (im1 - im2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
def readimage(path):

        result=imread(path)
        result = result.astype('float32')
        result/=255
        result=result[:,:,np.newaxis]
        return result
def Compression(path,path1,threshold=.5): #将路径中图像压缩还原并保存再路径中
    import cv2
    image=readimage(path)
    print(image.shape)
    minibatch =[image]
    minibatch=np.array(minibatch)
    print("$$$$$$")
    print(minibatch.shape)
    code, rec, code2, rec2, x= test(minibatch,threshold)
    print(code.shape)
    img=change(x)
    img2=change(rec2)
    print(img2.shape)
    print(img2.shape)
    cv2.imwrite(path1,img2)
    print(psnr(img,img2))

def load():
    com.load_weights('checkpoints/enc_mnist.npy')
    res.load_weights('checkpoints/dec_mnist.npy')

com,res = ComCNN(),ResCNN()
com.summary()
res.summary()
xt,xt1 = Mnist()



if __name__ == '__main__':
    test = get_trainer()
    get_session().run(ct.gvi())
    load()
    path = 'attack/1.bmp'
    print(path)
    path1 = 'defend/1.bmp'
    print(path1)
    Compression(path, path1)