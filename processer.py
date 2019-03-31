import numpy as np
import math
import cv2
import csv
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
        result = result.astype('float32')
        result/=255
        return result
def mergeimage_column(path,n):
    path1=path+str(n)+'.png'
    t1=cv2.imread(path1)
    for i in range(1,7):
        path2=path+str(n+i)+".png"
        t2=cv2.imread(path2)
        t1 = np.hstack((t1,t2))
    return t1
def mergeimage(path):
    t1=mergeimage_column(path,1)
    print(t1.shape)
    for  i in range(1,7):
         print(1+7*i)
         t2=mergeimage_column(path,1+7*i)
         t1 = np.vstack((t1,t2))
    return t1
def Divided_Pach(path1,path2):
    lena=cv2.imread(path1)
    height= 224
    width=224
    a=0
    b=0
    count=0
    print(width)
    print(height)
    path=path2
    while a<height:
            while b<width:
                box = lena[a:a+32,b:b+32]
                count=count+1
                b=b+32
                path2=path2+str(count)+".png"
                cv2.imwrite(path2,box)
                path2=path
            b=0
            a=a+32
    print(count)

def read_file(file_name):
    names=[]
    labels=[]
    csv_file=csv.reader(open(file_name,'r'))
    for stu in csv_file:
        names.append(stu[0])
        labels.append(stu[1])
    return names,labels