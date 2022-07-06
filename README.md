# Comdefend
The code for CVPR2019 (ComDefend: An Efficient Image Compression Model to Defend Adversarial Examples)
[paper](https://arxiv.org/abs/1811.12673)
The pure tensorflow of code is released in the [link](https://github.com/ZhangYuef/Comdefend_tensorflow)
## Environmental configuration
tensorflow>=1.1 </br>
python3 </br>
canton(pip install canton) </br>

## Train
Torch: Train_comdefend_Torch.py
TF: Train_comdefend_TF.py


## Description
**clean_image**: we select 7 clean images which are classified correctly by the classifier </br>
**attack_image:** we select 7 adversarial images which are attacked by the FGSM method </br>
**defend_image:** we use the Comdefend to deal with 7 adversarial images</br>
**chackpoints:** the model parameters </br>
**com_imagenet_temp, temp_imagenet:** the temporary files of the Comdefend</br>
**dev.csv:** correspondence between images and labels</br>
**Resnet_imagenet.py:** the classifier </br>
**compression_imagenet.py:** the Comdefend for Imagenet</br>
**compression_mnist.py:** the Comdefend for fashion_mnist</br>
## In addition
**E-mail:** jiaxiaojun@iie.ac.cn 
