# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = BasicConv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch, **kwargs):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            BasicConv2d(in_ch, in_ch,  kernel_size=3, padding=1, stride=2),
            BasicConv2d(in_ch, out_ch, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, side_ch=0,  **kwargs):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        
        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=3, padding=1, stride=2)
        #self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv = nn.Sequential(
            BasicConv2d(side_ch + in_ch, out_ch, kernel_size=3, padding=1),
            BasicConv2d(out_ch, out_ch, kernel_size=1, padding=0),
        )


    def forward(self, x1,  x2=None, refer_shape=None):
        
        x1 = self.up(x1)
        
        if x2 is not None:
            x1  =  match_tensor(x1, x2.size()[2::])
            x   =  torch.cat([x1, x2], dim=1)
        else:
            x = match_tensor(x1, refer_shape)

        x = self.conv(x)
        return x


class __down(nn.Module):
    def __init__(self, in_ch, out_ch, mode='135'):
        super(down, self).__init__()
        #self.mpconv = nn.Sequential(
        #    nn.MaxPool2d(2),
        #    double_conv(in_ch, out_ch)
        #)
        if mode == '135':
            self.mpconv = InceptionD_135_s2(in_ch, out_ch)
        elif mode == '13':
            self.mpconv = InceptionD_13_s2(in_ch, out_ch)

    def forward(self, x):
        x = self.mpconv(x)
        return x


class __up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, mode='135'):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        #self.conv = double_conv(in_ch, out_ch)
        if mode == '135':
            self.conv = InceptionA_135_s1(in_ch, out_ch)
        elif mode == '13':
            self.conv = InceptionA_13_s1(in_ch, out_ch)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if x2 is not None:
            x2  = match_tensor(x2, x1.size()[2::])
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)
        #self.conv = BasicConv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x


def match_tensor(out, refer_shape):
    
    skiprow,skipcol = refer_shape
    row, col = out.size()[2], out.size()[3]
    if skipcol >= col:
        pad_col = skipcol - col        
        left_pad_col  = pad_col // 2
        right_pad_col = pad_col - left_pad_col      
        out = F.pad(out, (left_pad_col, right_pad_col, 0,0), mode='reflect')
    else:
        crop_col = col - skipcol
        left_crop_col  = crop_col // 2
        right_col = left_crop_col + skipcol
        out = out[:,:,:, left_crop_col:right_col]
    
    if skiprow >= row:
        pad_row = skiprow - row
        left_pad_row  = pad_row // 2
        right_pad_row = pad_row - left_pad_row
        out = F.pad(out, (0,0, left_pad_row,right_pad_row), mode='reflect')
    else:
        crop_row = row - skiprow   
        left_crop_row  = crop_row // 2
        
        right_row = left_crop_row + skiprow
        
        out = out[:,:,left_crop_row:right_row, :]
    return out


class InceptionA_135_s1(nn.Module):
    '''feature 1,3,5, with stide 1'''
    def __init__(self, in_channels, out_channels,  
                 out_chan_1 = 64, out_chan_3 = 64, out_chan_5 = 64):

        super(InceptionA_135_s1, self).__init__()
        self.branch1x1   = BasicConv2d(in_channels, out_chan_1, kernel_size=1)

        self.branch5x5_2 = BasicConv2d(in_channels, out_chan_5,  kernel_size=(1, 5), padding=(0, 2))
        self.branch5x5_3 = BasicConv2d(out_chan_5, out_chan_5,  kernel_size=(5, 1), padding=(2, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, out_chan_3, kernel_size=3, padding=1)
        self.branch3x3dbl_2 = BasicConv2d(out_chan_3, out_chan_3,  kernel_size=3, padding=1)
        #self.branch3x3dbl_3 = BasicConv2d(out_chan_3, out_chan_3, kernel_size=3, padding=1)

        #self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)
        tmp_channels     = out_chan_1 + out_chan_3 + out_chan_5

        self.branch_out   = BasicConv2d(tmp_channels, out_channels, kernel_size=1)
        self.out_channels = out_channels

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        #branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(x)
        branch5x5 = self.branch5x5_3(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        #branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        #branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        #branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl]
        outputs = torch.cat(outputs, 1)
        outputs = self.branch_out(outputs)
        return outputs

class InceptionA_13_s1(nn.Module):
    '''feature 1,3,5, with stide 1'''
    def __init__(self, in_channels, out_channels, out_chan_1 = 96, out_chan_3 = 96):
        super(InceptionA_13_s1, self).__init__()
        self.branch1x1   = BasicConv2d(in_channels, out_chan_1, kernel_size=1)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, out_chan_3, kernel_size=3, padding=1)
        self.branch3x3dbl_2 = BasicConv2d(out_chan_3,  out_chan_3, kernel_size=3, padding=1)

        tmp_channels     = out_chan_1 + out_chan_3

        self.branch_out   = BasicConv2d(tmp_channels, out_channels, kernel_size=1)
        self.out_channels = out_channels

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        #branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        outputs = [branch1x1,  branch3x3dbl]
        outputs = torch.cat(outputs, 1)
        outputs = self.branch_out(outputs)
        return outputs

class InceptionD_135_s2(nn.Module):
    def __init__(self, in_channels, out_channels,
                 out_chan_3 = 96, out_chan_5 = 96):

        super(InceptionD_135_s2, self).__init__()
        #self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        #self.branch3x3_2 = BasicConv2d(192, out_chan_3, kernel_size=3, stride=2, padding = 1)
        self.branch3x3 = BasicConv2d(in_channels, out_chan_3, kernel_size=3, stride=2, padding = 1)
        
        #self.branch7x7x3_1 = BasicConv2d(in_channels, out_chan_5, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(in_channels, out_chan_5, kernel_size=(1, 5), padding=(0, 2))
        self.branch7x7x3_3 = BasicConv2d(out_chan_5, out_chan_5, kernel_size=(5, 1), padding=(2, 0))
        self.branch7x7x3_4 = BasicConv2d(out_chan_5, out_chan_5, kernel_size=3, stride=2, padding =1)
        
        tmp_channels = out_chan_3 + out_chan_5 + in_channels
        self.branch_out   = BasicConv2d(tmp_channels, out_channels, kernel_size=1)

        self.out_channels = out_channels

    def forward(self, x):
        
        branch3x3 = self.branch3x3(x)
        #branch3x3 = self.branch3x3_2(branch3x3)

        #branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(x)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        outputs = [branch3x3, branch7x7x3, branch_pool]

        outputs = torch.cat(outputs, 1)
        outputs = self.branch_out(outputs)

        return outputs

class InceptionD_13_s2(nn.Module):
    def __init__(self, in_channels, out_channels,
                 out_chan_3 = 128, out_chan_5 = 128):

        super(InceptionD_13_s2, self).__init__()
        #self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        #self.branch3x3_2 = BasicConv2d(192, out_chan_3, kernel_size=3, stride=2, padding = 1)
        self.branch3x3 = BasicConv2d(in_channels, out_chan_3, kernel_size=3, stride=2, padding = 1)
        
        #self.branch7x7x3_1 = BasicConv2d(in_channels, out_chan_5, kernel_size=1)
        #self.branch7x7x3_2 = BasicConv2d(out_chan_5, out_chan_5, kernel_size=(1, 3), padding=(0, 1))
        self.branch7x7x3_3 = BasicConv2d(in_channels, out_chan_5, kernel_size=3, padding=1)
        self.branch7x7x3_4 = BasicConv2d(out_chan_5, out_chan_5,  kernel_size=3, stride=2, padding =1)
        
        tmp_channels = out_chan_3 + out_chan_5 + in_channels
        self.branch_out   = BasicConv2d(tmp_channels, out_channels, kernel_size=1)

        self.out_channels = out_channels

    def forward(self, x):
        
        branch3x3 = self.branch3x3(x)
        #branch3x3 = self.branch3x3_2(branch3x3)

        #branch7x7x3 = self.branch7x7x3_1(x)
        #branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(x)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        outputs = [branch3x3, branch7x7x3, branch_pool]

        outputs = torch.cat(outputs, 1)
        outputs = self.branch_out(outputs)

        return outputs

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(  in_channels, out_channels, kernel_size=kernel_size, 
                                padding=padding, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.bn(x)
        return x