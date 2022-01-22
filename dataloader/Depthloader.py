import math
import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
import cv2
#import preprocess 
#import OpenEXR as exr

import numpy
import torch
import torchvision.transforms as transforms
import random

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

#__imagenet_stats = {'mean': [0.5, 0.5, 0.5],
#                   'std': [0.5, 0.5, 0.5]}

__imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}


def scale_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    #if scale_size != input_size:
    #t_list = [transforms.Scale((960,540))] + t_list

    return transforms.Compose(t_list)


def scale_random_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [
        transforms.RandomCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    if scale_size != input_size:
        t_list = [transforms.Scale(scale_size)] + t_list

    transforms.Compose(t_list)


def pad_random_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    padding = int((scale_size - input_size) / 2)
    return transforms.Compose([
        transforms.RandomCrop(input_size, padding=padding),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ])


def inception_preproccess(input_size, normalize=__imagenet_stats):
    return transforms.Compose([
        transforms.RandomSizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize)
    ])
def inception_color_preproccess(input_size, normalize=__imagenet_stats):
    return transforms.Compose([
        #transforms.RandomSizedCrop(input_size),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
        ),
        Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
        transforms.Normalize(**normalize)
    ])


def get_transform(name='imagenet', input_size=None,
                  scale_size=None, normalize=None, augment=True):
    normalize = __imagenet_stats
    input_size = 256
    if augment:
            return inception_color_preproccess(input_size, normalize=normalize)
    else:
            return scale_crop(input_size=input_size,
                              scale_size=scale_size, normalize=normalize)




class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class Grayscale(object):

    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class RandomOrder(object):
    """ Composes several transforms together in random order.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        if self.transforms is None:
            return img
        order = torch.randperm(len(self.transforms))
        for i in order:
            img = self.transforms[i](img)
        return img


class ColorJitter(RandomOrder):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.transforms = []
        if brightness != 0:
            self.transforms.append(Brightness(brightness))
        if contrast != 0:
            self.transforms.append(Contrast(contrast))
        if saturation != 0:
            self.transforms.append(Saturation(saturation))

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    a = Image.open(path).convert('RGB')

    return a

def depth_loader(filepath):
    img=cv2.imread(filepath,cv2.IMREAD_ANYDEPTH)
    img1 = img
    dst = cv2.inpaint(img,mask(img).astype('uint8'),3,cv2.INPAINT_NS)
    min_pixel = np.min(img1)
    img1[img1>1000] = 0
    max_pixel = np.max(img1)
    dst_new = np.clip(dst,min_pixel,max_pixel)
    return dst_new
def mask(new_img):
  im_result2 = np.zeros((256,512))
  for i in range(256):
    for j in range(512):
      if (new_img[i,j] > 1000.0):
        im_result2[i,j] = 1
      else:
        im_result2[i,j] = 0
  return im_result2

class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader= depth_loader):
 
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        disp_L= self.disp_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL = self.dploader(disp_L)


        if self.training:  

          # left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
          # right_img = right_img.crop((x1, y1, x1 + tw, y1 + th)) 
#           angle_y = np.array([(i - 0.5) / 256 * 360 for i in range(128, -128, -1)])
#           angle_ys = np.tile(angle_y[:, np.newaxis, np.newaxis], (1, 512, 1))
#           equi_info = angle_ys

#           angle_x = np.array([(i - 0.5) / 512 * 360 for i in range(256, -256, -1)])
#           angle_xs = np.tile(angle_x[np.newaxis,:, np.newaxis], (256, 1, 1))
#           equi_infox = angle_xs

           processed = get_transform(augment=False)
           left_img   = processed(left_img)
           right_img  = processed(right_img)
           left = np.array(left_img)
           right = np.array(right_img)
#           print(np.array(left_img).shape,np.array(right_img).shape)
#           equi_info = np.reshape(equi_info,(1,256,512))
#           equi_infox = np.reshape(equi_infox,(1,256,512))
#           left_img = np.concatenate([np.expand_dims(left, axis = 0), equi_infox, equi_info], 0)
#           right_img = np.concatenate([np.expand_dims(right, axis = 0), equi_infox, equi_info], 0)
           return left_img, right_img, dataL
        else:
           w, h = left_img.size

  
           w1, h1 = left_img.size

#           angle_y = np.array([(i - 0.5) / 256 * 360 for i in range(128, -128, -1)])
#           angle_ys = np.tile(angle_y[:, np.newaxis, np.newaxis], (1, 512, 1))
#           equi_info = angle_ys

#           angle_x = np.array([(i - 0.5) / 512 * 360 for i in range(256, -256, -1)])
#           angle_xs = np.tile(angle_x[np.newaxis,:, np.newaxis], (256, 1, 1))
#           equi_infox = angle_xs

           left = np.array(left_img)
           right = np.array(right_img)

#           equi_info = np.reshape(equi_info,(1,256,512))
#           equi_infox = np.reshape(equi_infox,(1,256,512))
#           left_img = np.concatenate([np.expand_dims(left, axis = 0), equi_infox, equi_info],0)
#           right_img = np.concatenate([np.expand_dims(right, axis = 0), equi_infox, equi_info],0) 


           processed = get_transform(augment=False)  
           left_img       = processed(left_img)
           right_img      = processed(right_img)

           return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)
