from __future__ import print_function
import argparse
from math import log10, sqrt
import os
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from dataloader import Newdepthloader as ls
from dataloader import Depthloader as DA
from models import *
parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
# parser.add_argument('--datapath', default='/media/jiaren/ImageNet/SceneFlowData/',
#                     help='datapath')
parser.add_argument('--datapathleft', default='/home/as04390/PSMNet-master-new',
                    help='datapathleft')
parser.add_argument('--datapathright', default='/home/as04390/PSMNet-master-new',
                    help='datapathright')
parser.add_argument('--epochs', type=int, default=1,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default=None,
                    help='load model')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
start_time = time.time()
all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp, val_left, val_right, val_depth = ls.dataloader(
    args.datapathleft, args.datapathright)
#print("Train",all_left_disp[10:20],"Test",test_left_disp[10:20])
TrainImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(
        all_left_img,all_right_img, all_left_disp, True),
    batch_size=1, shuffle=True, num_workers=4, drop_last=False)

Val = torch.utils.data.DataLoader(DA.myImageFloder(val_left, val_right, val_depth, False), batch_size=4, shuffle=False, num_workers=4, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
    batch_size=1, shuffle=False, num_workers=4, drop_last=False)
print("Time required toload data",time.time()-start_time)

if args.model == 'stackhourglass':
    model = stackhourglass()
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    print('Load pretrained model')
    pretrain_dict = torch.load(args.loadmodel)
    model.load_state_dict(pretrain_dict['state_dict'])

print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.001)


def PSNR(original,compressed):
    original = original
    compressed = compressed
    mse = np.mean((original - compressed) ** 2)

    if (mse == 0):
       return 0
    max_pixel = 1.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr




def train(imgL, imgR, depth_L):

    model.train()

    imgL, imgR, depth_true = imgL.cuda(), imgR.cuda(), depth_L.cuda()
#    print("training the images properly",disp_true.size())
    depth_true = torch.squeeze(depth_true,0)
    optimizer.zero_grad()

    if args.model == 'stackhourglass':
        output1, output2, output3 = model(imgL, imgR)
        output1 = torch.squeeze(output1)
        output2 = torch.squeeze(output2)
        output3 = torch.squeeze(output3)
        loss = 0.5*F.smooth_l1_loss(output1, depth_true) + 0.7*F.smooth_l1_loss(output2, depth_true) + F.smooth_l1_loss(output3, depth_true)
    elif args.model == 'basic':
        output = model(imgL, imgR)
        output = torch.squeeze(output, 1)
        loss = F.smooth_l1_loss(output, depth_true, size_average=True)

    loss.backward()
    optimizer.step()

    return loss.data


def test(imgL, imgR, depth_true):

    model.eval()
    imgL, imgR, depth_true = imgL.cuda(), imgR.cuda(), depth_true.cuda()
    depth_true = torch.tensor(depth_true,dtype = torch.float)
    depth_true = torch.squeeze(depth_true)

    # ----
    with torch.no_grad():
        output3 = model(imgL, imgR)
        output3 = torch.squeeze(output3)

    a = output3.cpu().numpy()

    a_new = a/np.max(a)
    b = depth_true.cpu().numpy()

    b_new = b/np.max(b)
    img = output3

    loss = F.smooth_l1_loss(img,depth_true)
    psnr = PSNR(b_new,a_new)
    return loss.data.cpu(),psnr


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    checkpoint = torch.load('./depth_new_checkpoint_1.tar')
    model.load_state_dict(checkpoint['state_dict'])

    optimizer.load_state_dict(checkpoint['optimizer_dict'])
    new_loss = checkpoint['val_loss']

    start_full_time = time.time()
    actual_loss = []
    for epoch in range(0, args.epochs):
        print('This is %d-th epoch' % (epoch))
        total_train_loss = 0

        count = 0
        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
            start_time = time.time()
#            if (np.max(np.asarray(disp_crop_L))<10000):
#              count = count+1
            loss = train(imgL_crop.float(), imgR_crop.float(), disp_crop_L.float())
            print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
            total_train_loss += loss
        print('epoch %d total training loss = %.3f' %
              (epoch, total_train_loss/len(TrainImgLoader)))
        print('count',count)
    # -----------------VAL---------------------------------------------------------
        total_val_loss = 0
        psnr_total = 0
        loss = []
#        loss.extend(new_loss)
        i = 0
        l = 0
        batch_loss = 0
        for batch, (imgL, imgR, disp_L) in enumerate(Val):
            val_loss,psnr = test(imgL.float(), imgR.float(), disp_L.float())
            actual_loss.append(val_loss)
            total_val_loss += val_loss
            psnr_total += psnr
            loss.append(val_loss)
#        fig = plt.figure()
#        plt.plot(np.linspace(1,len(loss),len(loss)),loss)
#        plt.xlabel("iterations")
#        plt.ylabel("Validation loss")
# changes are required here
#        fig.savefig('depth_checkpoint_new0.png', dpi=fig.dpi)
        print('total val loss = %.3f' % (total_val_loss/len(Val)))
        print("average psnr", (psnr_total/len(Val)))
     
# changes are required here
#        savefilename = args.savemodel+'/depth_new_checkpoint_6.tar'
#        torch.save({ 'epoch': epoch, 'state_dict': model.state_dict(),'optimizer_dict':optimizer.state_dict(), 'val_loss': loss}, savefilename)



#    fig = plt.figure()
#    plt.plot(np.linspace(1,len(actual_loss),len(actual_loss)),actual_loss)
#    plt.xlabel("iterations")
#    plt.ylabel("Validation loss")
#    fig.savefig('final_temp.png', dpi=fig.dpi)

#    print('full training time = %.2f HR' %
#          ((time.time() - start_full_time)/3600))
#    torch.save(model,"/home/as04390/PSMNet-master-new/model.pt")
    # ------------- TEST ------------------------------------------------------------
    total_test_loss = 0
    psnr_total = 0
    for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
        test_loss,psnr = test(imgL.float(), imgR.float(), disp_L.float())
        total_test_loss += test_loss
        psnr_total += psnr
    print('total test loss = %.3f' % (total_test_loss/len(TestImgLoader)))
    print("average test psnr",(psnr_total/len(TestImgLoader)))
    # ----------------------------------------------------------------------------------


if __name__ == '__main__':
    main()


