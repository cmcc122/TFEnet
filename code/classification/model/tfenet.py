import torch
import torch.nn as nn
import math
import scipy.io as io
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import math

class Residual_Block(nn.Module): 
    def __init__(self, inc, outc, dropout_rate):
        super(Residual_Block, self).__init__()
        
        if inc is not outc:
          self.conv_expand = nn.Conv2d(in_channels = inc, 
                                       out_channels = outc, 
                                       kernel_size = 1, 
                                       stride = 1, 
                                       padding = 0,
                                       bias = False)
        else:
          self.conv_expand = None          
        self.conv1 = nn.Conv2d(in_channels = inc, 
                               out_channels = outc, 
                               kernel_size = (1, 3), 
                               stride = 1, 
                               padding = (0, 1),
                               bias = False)
        self.bn1 = nn.BatchNorm2d(outc)
        self.conv2 = nn.Conv2d(in_channels = outc, 
                               out_channels = outc, 
                               kernel_size = (1, 3), 
                               stride = 1, 
                               padding = (0, 1),
                               bias = False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.dropout = nn.Dropout(dropout_rate)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain = 1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x): 
        if self.conv_expand is not None:
          identity_data = self.conv_expand(x)
        else:
          identity_data = x
        output = self.bn1(self.conv1(x))
        output = self.dropout(output)
        output = self.conv2(output)
        output = self.bn2(torch.add(output,identity_data))

       
        return output 
    
    
class Input_Layer(nn.Module):
    def __init__(self, inc,dropout_rate):
        super(Input_Layer, self).__init__()
        self.conv_input = nn.Conv2d(in_channels = inc, 
                                    out_channels = 4, 
                                    kernel_size = (1, 3), 
                                    stride = 1, 
                                    padding = (0, 1), 
                                    bias = False)
        self.bn_input = nn.BatchNorm2d(4)
        self.dropout = nn.Dropout(dropout_rate)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain = 1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        output = self.bn_input(self.conv_input(x))
        return output

    
def Embedding_Block(input_block, Residual_Block, num_of_layer, inc, outc,dropout_rate):
    layers = []
    layers.append(input_block(inc = inc,dropout_rate=dropout_rate))
    for i in range(0, num_of_layer):
        layers.append(Residual_Block(inc = int(math.pow(2, i)*outc), 
                                     outc = int(math.pow(2, i+1)*outc),
                                     dropout_rate=dropout_rate))
    return nn.Sequential(*layers) 

#分频
class MultiLevel_Spectral(nn.Module): 
    def __init__(self, inc, params_path='E:\data\code\classification\model\scaling_filter.mat'):
        super(MultiLevel_Spectral, self).__init__()
        self.filter_length = io.loadmat(params_path)['Lo_D'].shape[1]
        self.conv = nn.Conv2d(in_channels = inc, 
                              out_channels = inc*2, 
                              kernel_size = (1, self.filter_length), 
                              stride = (1, 2), padding = 0, 
                              groups = inc, 
                              bias = False)        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                f = io.loadmat(params_path)
                Lo_D, Hi_D = np.flip(f['Lo_D'], axis = 1).astype('float32'), np.flip(f['Hi_D'], axis = 1).astype('float32')
                m.weight.data = torch.from_numpy(np.concatenate((Lo_D, Hi_D), axis = 0)).unsqueeze(1).unsqueeze(1).repeat(inc, 1, 1, 1)            
                m.weight.requires_grad = False 
    
    def self_padding(self, x):
        return torch.cat((x[:, :, :, -(self.filter_length//2-1):], x, x[:, :, :, 0:(self.filter_length//2-1)]), (self.filter_length//2-1))
                           
    def forward(self, x): 
        out = self.conv(self.self_padding(x)) 
        return out[:, 0::2,:, :], out[:, 1::2, :, :]
   
class DepthwiseSeparableConv2D(nn.Module):
    def __init__(self, in_channels,out_channels, kernel_size, dropout_rate):
        super(DepthwiseSeparableConv2D, self).__init__()

        self.depthwise = nn.Sequential(nn.Conv2d(in_channels, in_channels, (1,kernel_size),
                                                 stride=(1, 1), padding=(0, kernel_size//2), groups=in_channels),
                                       nn.BatchNorm2d(in_channels),
                                       nn.LeakyReLU(0.01, inplace=True),
                                       )

        self.pointwise = nn.Sequential(nn.Conv2d(in_channels, out_channels, (1,1)),
                                       nn.BatchNorm2d(out_channels),
                                       nn.LeakyReLU(0.01, inplace=True),
                                        nn.Dropout(dropout_rate)
                                       )
        self.initialize()
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain = 1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    

class DAFM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels

        chunk_dim = dim // n_levels
        self.mfr = nn.ModuleList([nn.Conv2d(chunk_dim, chunk_dim, (1, 3), stride=1, padding=(0, 1), groups=chunk_dim) for i in range(self.n_levels)])
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act = nn.GELU()
                
    def forward(self, x):
        h, w = x.size()[-2:]
        xc = x.chunk(self.n_levels, dim=1)        
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h , w // 2**i)
                s = F.adaptive_max_pool2d(xc[i], p_size)   
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                s = self.mfr[i](xc[i])
            out.append(s)

        out = torch.cat(out, dim=1)
        out = self.aggr(out)
        out = self.act(out) * x
        return out
    
class EEG_Feature_Extraction(nn.Module):
    def __init__(self, inc, outc_max, kernel_size, dropout_rate):
        super(EEG_Feature_Extraction, self).__init__()
        self.conv0 = DepthwiseSeparableConv2D(inc,outc_max//4,kernel_size, dropout_rate=dropout_rate)
        
        self.dafm0=DAFM(dim=outc_max//4, n_levels=2)
        self.pooling0 = nn.MaxPool2d(kernel_size = (1,2),stride = (1, 2))

        self.conv1 = DepthwiseSeparableConv2D(outc_max//2,outc_max//4,kernel_size, dropout_rate=dropout_rate)
        
        self.dafm1=DAFM(dim=outc_max//4, n_levels=2)

        self.pooling1 = nn.MaxPool2d(kernel_size = (1, 2),stride = (1, 2))
        self.conv2 = DepthwiseSeparableConv2D(outc_max//4,outc_max//4,kernel_size, dropout_rate=dropout_rate)
        
        self.dafm2=DAFM(dim=outc_max//4, n_levels=2)

        self.pooling2 = nn.MaxPool2d(kernel_size = (1, 2),stride = (1, 2))
        self.conv3 = DepthwiseSeparableConv2D(outc_max//2,outc_max//4,kernel_size, dropout_rate=dropout_rate)
        
        self.dafm3=DAFM(dim=outc_max//4, n_levels=2)
        

    def forward(self, x):
        out = self.conv0(x)
        out = self.dafm0(out)
        out = self.pooling0(out)
        return out

class EFEM(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.11, map_reduce=8):#scale=0.1
        super(EFEM, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
        )

        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )
        self.ConvLinear = BasicConv(2 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain = 1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x0 = self.branch0(x)       
        x1 = self.branch1(x)
        out = 1.5*x0+x1
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)
        return out

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


 
def kernel_size(in_channel):
    k = int((math.log2(in_channel) + 1) // 2)  
    if k % 2 == 0:
        return k + 1
    else:
        return k
    
class TFEFM(nn.Module): 
    def __init__(self, in_channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.k = kernel_size(in_channel)
        self.channel_conv1 = nn.Conv1d(4, 1, kernel_size=self.k, padding=self.k // 2)
        self.channel_conv2 = nn.Conv1d(4, 1, kernel_size=self.k, padding=self.k // 2) 
        self.spatial_conv1 = nn.Conv2d(4, 1, kernel_size=7, padding=3)
        self.spatial_conv2 = nn.Conv2d(4, 1, kernel_size=7, padding=3)
        self.softmax = nn.Softmax(0)
        

    def forward(self, *inputs):
        
        min_len = min([x.shape[-1] for x in inputs])  
        reshape = nn.AdaptiveAvgPool2d((1, min_len))
        inputs = [reshape(x) for x in inputs]  
        t1 = sum(inputs[:-1]) 
        t2 = inputs[-1]

        # channel part   
        t1_channel_avg_pool = self.avg_pool(t1)  # b,c,1,1
        t1_channel_max_pool = self.max_pool(t1)  # b,c,1,1
        t2_channel_avg_pool = self.avg_pool(t2)  # b,c,1,1
        t2_channel_max_pool = self.max_pool(t2)  # b,c,1,1
        channel_pool = torch.cat([t1_channel_avg_pool, t1_channel_max_pool,
                                  t2_channel_avg_pool, t2_channel_max_pool],
                                 dim=2).squeeze(-1).transpose(1, 2)  # b,4,c
        t1_channel_attention = self.channel_conv1(channel_pool)  # b,1,c
        t2_channel_attention = self.channel_conv2(channel_pool)  # b,1,c
        
        channel_stack = torch.cat([t1_channel_attention.unsqueeze(0), 
                           t2_channel_attention.unsqueeze(0)], dim=0)
        channel_stack = self.softmax(channel_stack).transpose(-1, -2).unsqueeze(-1)  # 2,b,c,1,1
            
        # spatial part

        t1_spatial_avg_pool = torch.mean(t1, dim=1, keepdim=True)  # b,1,h,w
        t1_spatial_max_pool = torch.max(t1, dim=1, keepdim=True)[0]  # b,1,h,w
        t2_spatial_avg_pool = torch.mean(t2, dim=1, keepdim=True)  # b,1,h,w
        t2_spatial_max_pool = torch.max(t2, dim=1, keepdim=True)[0]  # b,1,h,w
        spatial_pool = torch.cat([t1_spatial_avg_pool, t1_spatial_max_pool,
                                  t2_spatial_avg_pool, t2_spatial_max_pool], dim=1)  # b,4,h,w
        t1_spatial_attention = self.spatial_conv1(spatial_pool)  # b,1,h,w
        t2_spatial_attention = self.spatial_conv2(spatial_pool)  # b,1,h,w
        spatial_stack = torch.cat([t1_spatial_attention.unsqueeze(0), 
                           t2_spatial_attention.unsqueeze(0)], dim=0)
        spatial_stack = self.softmax(spatial_stack)  # 2,b,1,h,w
        

        # fusion part, add 1 means residual add
        stack_attention = channel_stack + spatial_stack + 1  # 2,b,c,h,w#
        
        fuse = stack_attention[0] * t1 + stack_attention[1] * t2  # b,c,h,w
        return fuse
 
 
    

class Classification_Net(nn.Module):
    def __init__(self, inc, outc, dropout_rate):
        super(Classification_Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = inc, 
                               out_channels = inc, 
                               kernel_size = (1, 1), 
                               stride = 1, 
                               padding = (0, 0), 
                               bias = False)
        self.bn1 = nn.BatchNorm2d(inc)
        self.conv2 = nn.Conv2d(in_channels = inc, 
                               out_channels = outc, 
                               kernel_size = (1, 1), 
                               stride = 1, 
                               padding = (0, 0), 
                               bias = False)
        self.dropout = nn.Dropout(dropout_rate)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain = 1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        output1 = self.bn1(self.conv1(x))
        output1 = self.dropout(output1) 
        output = self.conv2(output1)
        output = self.dropout(output)  
        return output, output1

        
class TFEnet(nn.Module):
    def __init__(self, inc, class_num, si, outc_max = 64, num_of_layer = 1, dropout_rate=0.04):
        super(TFEnet, self).__init__()  
        self.fi = math.floor(math.log2(si))
        self.embedding = Embedding_Block(Input_Layer, 
                                         Residual_Block, 
                                         num_of_layer = num_of_layer, 
                                         inc = inc, 
                                         outc = 4,
                                         dropout_rate=dropout_rate) 
        self.MultiLevel_Spectral = MultiLevel_Spectral(inc = inc)

        self.gamma_x = EEG_Feature_Extraction(inc = (4*int(math.pow(2, num_of_layer))+inc)*2, outc_max = outc_max, kernel_size = 7, dropout_rate=dropout_rate)
        self.beta_x = EEG_Feature_Extraction(inc = inc, outc_max = outc_max, kernel_size = 7, dropout_rate=dropout_rate)
        self.alpha_x = EEG_Feature_Extraction(inc = inc, outc_max = outc_max, kernel_size = 3, dropout_rate=dropout_rate)
        self.theta_x = EEG_Feature_Extraction(inc = inc, outc_max = outc_max, kernel_size = 3, dropout_rate=dropout_rate)
        self.embedding_x=EEG_Feature_Extraction(inc = 4*int(math.pow(2, num_of_layer))+inc, outc_max = outc_max, kernel_size = 3, dropout_rate=dropout_rate)#原kz3
        self.delta_x = EEG_Feature_Extraction(inc = (4*int(math.pow(2, num_of_layer))+inc)*2, outc_max = outc_max, kernel_size = 3, dropout_rate=dropout_rate)    
        self.efem_x = EFEM(in_planes=outc_max//4, out_planes=outc_max//4)   

        self.reshape = nn.AdaptiveAvgPool2d(1)
        self.tfefm =  TFEFM(in_channel=outc_max//4)
        self.classifier = Classification_Net(inc = outc_max//4, outc = class_num,dropout_rate=dropout_rate)
        
    def forward(self, x): 
        embedding_x = self.embedding(x) 
        cat_x = torch.cat((embedding_x, x), 1)

        for i in range(1, self.fi-2):
            if i <= self.fi-7:
                if i == 1:
                    out, _ = self.MultiLevel_Spectral(x)
                else:
                    out, _ = self.MultiLevel_Spectral(out)
            elif i == self.fi-6:
                if self.fi >= 8:
                    out, gamma = self.MultiLevel_Spectral(out)
                else:
                    out, gamma = self.MultiLevel_Spectral(x)
            elif i == self.fi-5:
                out, beta = self.MultiLevel_Spectral(out)
            elif i == self.fi-4:
                out, alpha = self.MultiLevel_Spectral(out)
            elif i == self.fi-3:
                delta, theta = self.MultiLevel_Spectral(out)
        
        x2 = self.beta_x(beta)
        x3 = self.alpha_x(alpha)
        x4 = self.theta_x(theta)

        x6 = self.embedding_x(cat_x)
        

        x2 = self.efem_x(x2)
        x3 = self.efem_x(x3)
        x4 = self.efem_x(x4)

        x6 = self.efem_x(x6)

        #特征融合
        mix_x=self.tfefm(x2,x3,x4,x6)
        mix_x=self.reshape(mix_x)
        output, decov1= self.classifier(mix_x)

        return output.squeeze()



if __name__ == "__main__":
    from thop import profile
    x = Variable(torch.ones([32, 30, 1, 512]))
    model = TFEnet(30, 2, 128)
    # x = Variable(torch.ones([32, 61, 1, 128]))
    # model = CE_stSENet(61, 2, 128)
    flops, params = profile(model, inputs=(x,))
    print(f"FLOPs: {flops}, Params: {params}")

    output = model(x)
    print(output.shape)
    
