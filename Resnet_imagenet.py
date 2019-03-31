import foolbox
import torch
import torchvision.models as models
import numpy as np
import cv2
import torchvision.transforms as transforms
from processer import *
# instantiate the model
resnet101 = models.resnet101(pretrained=True).eval()
if torch.cuda.is_available():
    resnet101 = resnet101.cuda()
mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
fmodel = foolbox.models.PyTorchModel(
    resnet101, bounds=(0, 1), num_classes=1000, preprocessing=(mean, std))
names,labels=read_file('dev.csv')
count=0


path='clean_image/'+names[0]
print(path)
img=cv2.imread(path)
img=cv2.resize(img,(224,224))
transform = transforms.Compose([transforms.ToTensor()])
img = transform(img).numpy()
label=int(labels[0])
label=np.array(label)
label=label.astype(np.int64)

print('label', label)
print('predicted class', np.argmax(fmodel.predictions(img)))

