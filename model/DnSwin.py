import torch
from torch import nn
from model.S_T import SwinTransformer

def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1) #(B,C*4,H/2,W/2)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r**2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False 

    def forward(self, x):
        return dwt_init(x)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

class RB(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(RB, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope, inplace=True),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope, inplace=True))

        self.conv = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        c0 = self.conv(x)
        x = self.block(x)
        return x + c0

class NRB(nn.Module):
    def __init__(self, n, in_size, out_size, relu_slope):
        super(NRB, self).__init__()
        nets = []
        for i in range(n):
            nets.append(RB(in_size, out_size, relu_slope))
        self.body = nn.Sequential(*nets)

    def forward(self, x):
        return self.body(x)

class Network(nn.Module):
    def __init__(self, in_chn=3, wf=16, relu_slope=0.2):
        super(Network, self).__init__()        

        self.l1 = RB(in_size=in_chn, out_size=wf, relu_slope=relu_slope)
        self.down1 = DWT()

        self.l2 = RB(in_size=wf*4, out_size=wf*4, relu_slope=relu_slope)
        self.down2 = DWT()

        self.l3 = RB(in_size=wf*16, out_size=wf*16, relu_slope=relu_slope)
        self.down3 = DWT()
        
        self.l4 = RB(in_size=wf*64, out_size=wf*64, relu_slope=relu_slope)
        
        self.up3 = IWT()
        self.u3 = RB(in_size=wf*32, out_size=wf*16, relu_slope=relu_slope)

        self.up2 = IWT()
        self.u2 = RB(in_size=wf*8, out_size=wf*4, relu_slope=relu_slope)

        self.up1 = IWT()
        self.u1 = RB(in_size=wf*2, out_size=wf, relu_slope=relu_slope)
        
        #skip phase                         
        self.st_1 = nn.Sequential(
            NRB(3, in_size=wf, out_size=wf, relu_slope=relu_slope),
            SwinTransformer(pretrain_img_size=128,patch_size=1,in_chans=16,embed_dim=96,depths=[2],num_heads=[3],window_size=8)
        )
        self.st_2 = nn.Sequential(
            NRB(2, in_size=wf*4, out_size=wf*4, relu_slope=relu_slope),
            SwinTransformer(pretrain_img_size=64,patch_size=1,in_chans=64,embed_dim=192,depths=[2],num_heads=[6],window_size=8)
        )
        self.st_3 = nn.Sequential(
            NRB(1, in_size=wf*16, out_size=wf*16, relu_slope=relu_slope),
            SwinTransformer(pretrain_img_size=32,patch_size=1,in_chans=256,embed_dim=768,depths=[4],num_heads=[6],window_size=8)
        )
        self.st_4 = SwinTransformer(pretrain_img_size=16,patch_size=1,in_chans=1024,embed_dim=1536,depths=[4],num_heads=[6],window_size=8)

        self.last = conv3x3(wf, in_chn, bias=True)

    def forward(self, x1):
        o1 = self.l1(x1) # 16,128,128
        d1 = self.down1(o1) #64,64,64

        o2 = self.l2(d1) #64,64,64
        d2 = self.down2(o2) #256,32,32

        o3 = self.l3(d2) #256,32,32
        d3 = self.down3(o3) #1024,16,16

        o4 = self.l4(d3) #1024,16,16
        o4 = self.st_4(o4)

        u3 = self.up3(o4) #256,32,32
        u3 = torch.cat([u3, self.st_3(o3)], dim=1) #512,32,32
        u3 = self.u3(u3) #256,32,32

        u2 = self.up2(u3) #64,64,64
        u2 = torch.cat([u2, self.st_2(o2)], dim=1) #128,64,64
        u2 = self.u2(u2) #64,64,64

        u1 = self.up1(u2) #16,128,128
        u1 = torch.cat([u1, self.st_1(o1)], dim=1) #32,128,128
        u1 = self.u1(u1) #16,128,128

        out = self.last(u1) #3,128,128
        return out

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)
