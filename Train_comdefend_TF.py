import sys
import numpy as np
import canton as ct
from canton import *
import tensorflow as tf
import time
import os
import math
def con10(num):
    list=[]
    i=0
    while(num!=0):
        list.insert(0,int(num%2))
        num=int(num/2)
    len1=len(list)
    if len1==32:
        return list
    else:
        remain=32-len1
        for i in range(remain):
           list.insert(0,0)
        return list

def cifar():
    from keras.datasets import cifar10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()#32*32的数据集

    print('X_train shape:', X_train.shape)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255
    return X_train,X_test

def encoder():
    c=Can()
    def conv(nip,nop,tail=True):
        c.add(Conv2D(nip,nop,k=3,usebias=True))
        if tail:
            # c.add(BatchNorm(nop))
            c.add(Act('elu'))
    c.add(Lambda(lambda x:x-0.5))
    conv(3,16)
    conv(16,32)
    conv(32,64)
    conv(64,128)
    conv(128,256)
    conv(256,128)
    conv(128,64)
    conv(64,32)
    conv(32,12,tail=False)
    c.chain()
    return c

def decoder():
    c=Can()
    def conv(nip,nop,tail=True):
        c.add(Conv2D(nip,nop,k=3,usebias=True))
        if tail:
            # c.add(BatchNorm(nop))
            c.add(Act('elu'))
    conv(12,32)
    conv(32,64)
    conv(64,128)
    conv(128,256)
    conv(256,128)
    conv(128,64)
    conv(64,32)
    conv(32,16)
    conv(16,3,tail=False)
    c.add(Act('sigmoid'))
    c.chain()
    return c

def get_trainer():
    x = ph([None,None,3])

    # augment the training set by adding random gain and bias pertubation
    sx = tf.shape(x)

    noisy_x = x
    noisy_x = tf.clip_by_value(noisy_x,clip_value_max=1.,clip_value_min=0.)

    code_noise = tf.Variable(1.0)
    linear_code = enc(noisy_x)

    # add gaussian before sigmoid to encourage binary code
    noisy_code = linear_code - \
        tf.random_normal(stddev=code_noise,shape=tf.shape(linear_code))
    binary_code = Act('sigmoid')(noisy_code)

    y = dec(binary_code)
    loss = tf.reduce_mean((y-noisy_x)**2) + tf.reduce_mean(binary_code**2) * 0.0001

    opt = tf.train.AdamOptimizer()
    train_step = opt.minimize(loss,
        var_list=enc.get_weights()+dec.get_weights())

    def feed(batch,cnoise):
        sess = ct.get_session()
        res = sess.run([train_step,loss],feed_dict={
            x:batch,
            code_noise:cnoise,
        })
        return res[1]

    set_training_state(False)
    quantization_thresholcomdefendd = tf.Variable(0.5)
    binary_code_test = tf.cast(binary_code>quantization_threshold,tf.float32)
    y_test = dec(binary_code_test)

    def test(batch,quanth):
        sess = ct.get_session()
        res = sess.run([binary_code_test,y_test,binary_code,y,noisy_x,x],feed_dict={
            x:batch,
            quantization_threshold:quanth,
        })
        return res
    return feed,test










def r(ep=1,cnoise=0.1):
    np.random.shuffle(xt)
    length = len(xt)
    bs = 20
    for i in range(ep):
        print('ep',i)
        for j in range(0,length,bs):
            minibatch = xt[j:j+bs]
            loss = feed(minibatch,cnoise)
            print(j,'loss:',loss)
        if j % 1000 == 0:
                show()

def show(threshold=.5):
    bs = 1

    j = np.random.choice(len(xt1)-1)
    minibatch = xt[j:j+bs]

    code, rec, code2, rec2, noisy_x,x = test(minibatch,threshold)
    print(code.shape)
    print('******')
    img=change(x)
    img2=change(rec)
    print(psnr(img,img2))
    temp_psnr=psnr(img,img2)
    print(temp_psnr)
    return temp_psnr


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

        result=cv2.imread(path)
        img=[cv2.imread(path)][0]
        cv2.imwrite('raw2.jpg',img)
        result = result.astype('float32')
        result/=255
        return result
def Compression(path,path1,threshold=.5): #将路径中图像压缩还原并保存再路径中 code bool code2 float
    import cv2
    image=readimage(path)
    minibatch =[image]
    minibatch=np.array(minibatch)
    print(minibatch.shape)
    code, rec, code2, rec2, noisy_x,x= test(minibatch,threshold)
    img=change(x)

    img1=change(rec2)
    cv2.imwrite('raw.jpg',img1)
    img2=change(rec)
    cv2.imwrite(path1,img2)
    print(psnr(img,img2))
    return code, code2


def save():
    enc.save_weights('enc.npy')
    dec.save_weights('dec.npy')

def load():
    enc.load_weights('enc/enc20_0.0001.npy')
    dec.load_weights('dec/dec20_0.0001.npy')

enc,dec = encoder(),decoder()
enc.summary()
dec.summary()
xt,xt1 = cifar()

def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        print(files)  # 当前路径下所有非目录子文件
    return files



if __name__ == '__main__':
    import cv2
    feed,test = get_trainer()
    get_session().run(ct.gvi())
    r(ep=1, cnoise=20.0)
    save()

    #     cpath='image_temp/ubool/'+str(i)+'.bmp'
    #     cv2.imwrite(cpath,code[0][:,:,i]*255)
    #     cpath1='image_temp/ufloat/'+str(i)+'.bmp'
    #     cv2.imwrite(cpath1,code2[0][:,:,i]*255)
