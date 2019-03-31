import sys
from processer import *
import numpy as np
import canton as ct
from canton import *
import tensorflow as tf
import time
import os
import math
import cv2
def mkdir(path):
    # 引入模块
    import os
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print('make')
def ComCNN():
    c=Can()
    def conv(nip,nop,flag=True):
        c.add(Conv2D(nip,nop,k=3,usebias=True))
        if flag:
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
    conv(32,12,flag=False)
    c.chain()
    return c

def RecCNN():
    c=Can()
    def conv(nip,nop,flag=True):
        c.add(Conv2D(nip,nop,k=3,usebias=True))
        if flag:
            c.add(Act('elu'))
    conv(12,32)
    conv(32,64)
    conv(64,128)
    conv(128,256)
    conv(256,128)
    conv(128,64)
    conv(64,32)
    conv(32,16)
    conv(16,3,flag=False)
    c.add(Act('sigmoid'))
    c.chain()
    return c

def get_defense():
    x = ph([None,None,3])
    x = tf.clip_by_value(x,clip_value_max=1.,clip_value_min=0.)
    code_noise = tf.Variable(1.0)
    linear_code = com(x)
    noisy_code = linear_code - \
        tf.random_normal(stddev=code_noise,shape=tf.shape(linear_code))
    binary_code = Act('sigmoid')(noisy_code)
    y = rec(binary_code)
    set_training_state(False)
    quantization_threshold = tf.Variable(0.5)
    binary_code_test = tf.cast(binary_code>quantization_threshold,tf.float32)
    y_test = rec(binary_code_test)
    def test(batch,quanth):
        sess = ct.get_session()
        res = sess.run([binary_code_test,y_test,binary_code,y,x],feed_dict={
            x:batch,
            quantization_threshold:quanth,
        })
        return res
    return test

def Compression(path,path1,threshold=.5): #将路径中图像压缩还原并保存再路径中 code bool code2 float
    import cv2
    image=readimage(path)
    minibatch =[image]
    minibatch=np.array(minibatch)
    #print(minibatch.shape)
    code, rec, code2, rec2,x= test(minibatch,threshold)
    img2=change(rec)
    cv2.imwrite(path1,img2)

    return code, code2
def load():
    print("****")
    com.load_weights('checkpoints/enc20_0.0001.npy')
    rec.load_weights('checkpoins/dec20_0.0001.npy')
com,rec = ComCNN(),RecCNN()
com.summary()
rec.summary()
import time
if __name__ == '__main__':
    test = get_defense()
    get_session().run(ct.gvi())
    load()
    start = time.time()
    path='adv.png'
    path3="temp_imagenet/"
    mkdir(path3)
    Divided_Pach(path,path3)
    for i in range(1,50):
            mkdir("temp_imagenet/")
            mkdir("com_imagenet_temp/")
            path1="temp_imagenet/"+str(i)+'.png'
            path2="com_imagenet_temp/"+str(i)+'.png'
            print(path2)
            Compression(path1,path2)
    concated=mergeimage('com_imagenet_temp/')
        #print(concated.shape)
    path4='result.png'
    print(path4)
    cv2.imwrite(path4,concated)
    end = time.time()
print(end-start)