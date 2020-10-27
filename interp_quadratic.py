"""
Xiangyu Xu, Siyao Li, Wenxiu Sun, Qian Yin, and Ming-Hsuan Yang, "Quadratic Video Interpolation", NeurIPS 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .acceleration import AcFusionLayer as Acceleration
from .acceleration import CalAcc
from .flow_reversal import FlowReversal
from .UNet2 import UNet2 as UNet
from .PWCNetnew import PWCNet
#from .visualization import visacc, make_color_wheel, compute_color, flow_to_image

import sys



#import matplotlib.pyplot as plt

#def sparse_flow(flow, index, X=None, Y=None, stride=1):
#    flow = flow.copy()
#    #flow[:,:,0] = -flow[:,:,0]
#    flow[0, 0, :, :] = -flow[0, 0, :, :]
#    if X is None:
#        #height, width, _ = flow.shape
#        _, _, height, width = flow.shape
#        print("height:{}".format(height))
#        print("width:{}".format(width))
#
#        xx = np.arange(0,height,stride)
#        yy = np.arange(0,width,stride)
#        X, Y= np.meshgrid(xx,yy)
#        X = X.flatten()
#        Y = Y.flatten()

        # sample
#        sample_0 = flow[0, 0, :, :][xx]
#        sample_0 = sample_0.T
#        sample_x = sample_0[yy]
#        sample_x = sample_x.T
#        sample_1 = flow[0, 1, :, :][xx]
#        sample_1 = sample_1.T
#        sample_y = sample_1[yy]
#        sample_y = sample_y.T

#        sample_x = sample_x[0, np.newaxis, :, :]
#        sample_y = sample_y[0,np.newaxis, :, :]
#        new_flow = np.concatenate([sample_x, sample_y], axis=2)
#    flow_x = new_flow[0, 0, :, :].flatten()
#    flow_y = new_flow[0, 1, :, :].flatten()

#    filename = "./acc-pics/" + str(index) + ".png"

#cv2.imsave(filename, )
    
    # display
#    ax = plt.gca()
#    ax.xaxis.set_ticks_position('top')
#    ax.invert_yaxis()
    # plt.quiver(X,Y, flow_x, flow_y, angles="xy", color="#666666")
#    ax.quiver(X,Y, flow_x, flow_y, color="#666666")
#    ax.grid()
    # ax.legend()
#    plt.draw()
    #plt.show()
#    plt.savefig("./acc-pics/" + index + ".jpg")
#:w

#print("a")




   














def backwarp(img, flow):
    _, _, H, W = img.size()

    u = flow[:, 0, :, :]
    v = flow[:, 1, :, :]

    gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))

    gridX = torch.tensor(gridX, requires_grad=False,).cuda()
    gridY = torch.tensor(gridY, requires_grad=False,).cuda()
    x = gridX.unsqueeze(0).expand_as(u).float() + u
    y = gridY.unsqueeze(0).expand_as(v).float() + v
    # range -1 to 1
    x = 2*(x/W - 0.5)
    y = 2*(y/H - 0.5)
    # stacking X and Y
    grid = torch.stack((x,y), dim=3)
    # Sample pixels using bilinear interpolation.
    imgOut = torch.nn.functional.grid_sample(img, grid)

    return imgOut


class SmallMaskNet(nn.Module):
    """A three-layer network for predicting mask"""
    def __init__(self, input, output):
        super(SmallMaskNet, self).__init__()
        self.conv1 = nn.Conv2d(input, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, output, 3, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        x = self.conv3(x)
        return x


class QVI(nn.Module):
    """Quadratic Video Interpolation"""
    def __init__(self, path='./network-default.pytorch'):
        super(QVI, self).__init__()
        self.flownet = PWCNet()
        self.acc = Acceleration()
        self.fwarp = FlowReversal()
        self.refinenet = UNet(20, 8)
        self.masknet = SmallMaskNet(38, 1)
        self.cal_acc = CalAcc()

        self.flownet.load_state_dict(torch.load(path))

    def forward(self, I0, I1, I2, I3, t, index):
    #def forward(self, I0, I1, I2, I3, t):

        # Input: I0-I3: (N, C, H, W)
        #          t: inter-time (N, 1, 1, 1) or constant*
        if I0 is not None:
            F10 = self.flownet(I1, I0).float()
        else:
            F10 = None

        F12 = self.flownet(I1, I2).float()
        F21 = self.flownet(I2, I1).float()

        if I3 is not None:
            F23 = self.flownet(I2, I3).float()
        else:
            F23 = None

        if F10 is not None and F23 is not None:
            F1ta, F2ta = self.acc(F10, F12, F21, F23, t)
                       
            F1t = F1ta
            F2t = F2ta

            Fa = self.cal_acc(F10, F12, F21, F23,  t)

            #enc10 = F10.detach().cpu().numpy()
            enc12 = F12.detach().cpu().numpy()
            enca = Fa.detach().cpu().numpy()

            np.savez("/home/thu-skyworks/dengzh18/Transform/nips/npzs/npz_12/" + str(index).zfill(5) + ".png", enc12)
            np.savez("/home/thu-skyworks/dengzh18/Transform/nips/npzs/npz_a/" + str(index).zfill(5) + ".png", enca)
            print("enc")

            #AC = self.cal_acc(F10, F12, F21, F23,  t)
            #AC = self.cal_acc(F10, F12, t)
            #AC = F1t + F2t
            #enc = F1t.detach().cpu().numpy()
            #np.savez("./kick_1/kick_1_flow/" + str(index) + ".npz", enc)
            print("ac-enc")
            #print(AC.shape)
            
            #a1, a2, ac, ad = self.cal_acc(F10, F12, F21, F23, t)
            #enc1 = a1.detach().cpu().numpy()
            #enc2 = a2.detach().cpu().numpy()
            #encc = ac.detach().cpu().numpy()
            #encd = ad.detach().cpu().numpy()
            #np.savez("./npzs/acc1/" + str(index) + ".npz", enc1)
            #np.savez("./npzs/acc2/" + str(index) + ".npz", enc2)
            #np.savez("./npzs/accc/" + str(index) + ".npz", encc)
            #np.savez("./npzs/accd/" + str(index) + ".npz", encd)
            #print("end")

            #ac = self.cal_acc(F10, F12, F21, F23, t)
            #enc = ac.detach().cpu().numpy()
            #np.savez("./kick_1/kick_1_ac/" + str(index) + ".npz", enc)
            #print("ac")


        else:
            print("ELSE here")
            F1t = t * F12
            F2t = (1-t) * F21

        #print(F1t.shape)
        #visacc(F1t, index)
        #sparse_flow(F1t, index)
        #print(index)

        #filename = "./acc-pics/" + str(index) + "_a.txt"
        #a = np.array(F1t[0, 0, :, :].clone().cpu()[0])
        #np.savetxt(filename, a)
        #filename = "./acc-pics/" + str(index) + "_b.txt"
        #b = np.array(F1t[0, 1, :, :].clone().cpu()[0])
        #np.savetxt(filename, b)
        #print("SAVED")

        #test print
        #a = F1t[0, 1, :, :]
        #print(a.shape)
        #print(index)
        #img = flow_to_image_1(F1t)
        #m = index
        #cv2.imwrite("./acc-pics/"  + ".png", img)
        #print("WRITTEN!")

        #enc = AC.detach().cpu().numpy()
        #np.savez("./flow_pics/" + str(index) + ".npz", enc)
        #print("enc")

        # Flow Reversal
        Ft1, norm1 = self.fwarp(F1t, F1t)
        Ft1 = -Ft1
        Ft2, norm2 = self.fwarp(F2t, F2t)
        Ft2 = -Ft2

        Ft1[norm1 > 0] = Ft1[norm1 > 0]/norm1[norm1>0].clone()
        Ft2[norm2 > 0] = Ft2[norm2 > 0]/norm2[norm2>0].clone()


        I1t = backwarp(I1, Ft1)
        I2t = backwarp(I2, Ft2)

        output, feature = self.refinenet(torch.cat([I1, I2, I1t, I2t, F12, F21, Ft1, Ft2], dim=1))

        # Adaptive filtering
        Ft1r = backwarp(Ft1, 10*torch.tanh(output[:, 4:6])) + output[:, :2]
        Ft2r = backwarp(Ft2, 10*torch.tanh(output[:, 6:8])) + output[:, 2:4]

        I1tf = backwarp(I1, Ft1r)
        I2tf = backwarp(I2, Ft2r)

        M = torch.sigmoid(self.masknet(torch.cat([I1tf, I2tf, feature], dim=1))).repeat(1, 3, 1, 1)

        It_warp = ((1-t) * M * I1tf + t * (1 - M) * I2tf) / ((1-t) * M + t * (1-M)).clone()

        return It_warp




