from __future__ import print_function
import argparse
import math
import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
import math
from models import *
import cv2
from PIL import Image
import matplotlib.pyplot as plt
# 2012 data /media/jiaren/ImageNet/data_scene_flow_2012/testing/

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/testing/',
                    help='select model')
parser.add_argument('--loadmodel', default='./newmodel.pt',
                    help='loading model')
parser.add_argument('--leftimg', default= './train_left.png',
                    help='load model')
parser.add_argument('--rightimg', default= './train_right.png',
                    help='load model')                                      
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.model == 'stackhourglass':
    model = stackhourglass()
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    checkpoint = torch.load('./depth_new_checkpoint_3.tar')
    model.load_state_dict(checkpoint['state_dict'])


print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
def mask(new_img):
  im_result2 = np.zeros((256,512))
  for i in range(256):
    for j in range(512):
      if (new_img[i,j] > 1000.0):
        im_result2[i,j] = 1
      else:
        im_result2[i,j] = 0
  return im_result2



def test(imgL,imgR):
        model.eval()


        with torch.no_grad():
            disp = model(imgL,imgR)
        disp = torch.squeeze(disp)
#        print("disp",disp) 
        pred_disp = disp.data.cpu().numpy()
#        print("pred_disp",pred_disp) 
#        for i in range(512):
#          for j in range(256):
#            if pred_disp[j,i] == 0:
#              pred_disp[j,i] = pred_disp[j,511]
#        b = 0.26
#        angle = np.zeros((512,256))
#        for i in range(256):
#          for j in range(512):
#            theta_T = ((j) * (2 * math.pi)/ 512)
#            angle[j,i] = b* math.sin(theta_T) 
#        new_angle = angle.T
#        new_sin = np.zeros((256,512))
#        for i in range(512):
#          for j in range(256):
#            new_sin[j,i] = math.sin(pred_disp[j,i])


#        a = np.divide(new_angle,new_sin)
#        a_new = a/np.max(a)
#        depth = (0.1965 * 260.0)/pred_disp
        return pred_disp

def PSNR(original,compressed):
    original = original
    compressed = compressed
    mse = np.mean((original - compressed) ** 2)

    if (mse == 0):
       return 0
#    print(np.max(original),np.max(compressed))
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def main():

        normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                            'std': [0.229, 0.224, 0.225]}
        infer_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(**normal_mean_var)])    

        imgL_o = Image.open(args.leftimg).convert('RGB')
        imgR_o = Image.open(args.rightimg).convert('RGB')
        depth = cv2.imread("./train_depth.exr", cv2.IMREAD_ANYDEPTH)
        img1 = depth
        dst = cv2.inpaint(depth,mask(depth).astype('uint8'),3,cv2.INPAINT_NS)
        min_pixel = np.min(img1)
        img1[img1>1000] = 0
        max_pixel = np.max(img1)
        print(min_pixel,max_pixel)
        dst_new = np.clip(dst,min_pixel,max_pixel)
        depth = dst_new
        imgL = infer_transform(imgL_o)
        imgR = infer_transform(imgR_o) 

        imgL = imgL.unsqueeze(0)
        imgR = imgR.unsqueeze(0)
        start_time = time.time()
        pred_disp = test(imgL,imgR)
        print('time = %.2f' %(time.time() - start_time))
        print(PSNR(depth/np.max(depth),pred_disp/np.max(pred_disp)))

        img = pred_disp
        new_img = np.copy(img)
        max = np.max(new_img)
        print(max,np.min(new_img))

        new_img = (pred_disp*255).astype('uint16')
        img = Image.fromarray(new_img)
        f = plt.figure()
        plt.imshow(img,cmap = 'gray')
        f.savefig('test_train_imageinpaint3.png',dpi = f.dpi)

if __name__ == '__main__':
   main()






