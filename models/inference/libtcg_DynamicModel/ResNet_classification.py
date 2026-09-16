import torch
import json
import os

from torch import nn
from pathlib import Path
# from torchsummary import summary

class Block(nn.Module):
    def __init__(self, inp_channels = 64, out_channels = 64, is_down = False):
        super().__init__()
        
        self.conv1 = nn.Conv2d(inp_channels,
                               out_channels,
                               kernel_size = 3,
                               stride = 2 if is_down else 1,
                               padding = 1,
                               bias = False)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size = 3,
                               stride = 1,
                               padding = 1,
                               bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.is_down = is_down
        self.down = nn.Sequential(
            nn.Conv2d(inp_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        x0 = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.is_down:
            x0 = self.down(x0)
            
        x = x + x0
        x = self.relu(x)
        
        return x

class CNN2D_embed(nn.Module):
    def __init__(self,
                 press_levels = 1,
                 encode_channels=1):
        super().__init__()
        self.relu = nn.ReLU(True)
        
        self.conv1 = nn.Conv2d(in_channels = press_levels,
                               out_channels = 32,
                               kernel_size = 3,
                               padding = 1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(in_channels = 32,
                               out_channels = 32,
                               kernel_size = 3,
                               padding = 1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(in_channels = 32,
                               out_channels = encode_channels,
                               kernel_size = 3,
                               padding = 1)
        self.bn3 = nn.BatchNorm2d(encode_channels)
        
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        return x


class Resnet(nn.Module):
    def __init__(self, 
                 inp_channels, 
                 num_residual_block = [2, 2, 2, 2], 
                 num_class = 1,
                 vector = False):
        super().__init__()
        
        self.vector = vector
        self.first_channels = 64
        self.conv1 = nn.Conv2d(inp_channels,
                               out_channels = self.first_channels,
                               kernel_size = 7,
                               stride = 2,
                               padding = 3,
                               bias = False)
        self.bn1 = nn.BatchNorm2d(self.first_channels)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,
                                    stride = 2,
                                    padding = 1,)
        self.resnet, out_channels = self.resnet_layer(num_residual_block)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features = out_channels,
                            out_features = num_class,
                            bias = True)
        self.sm = nn.Softmax(dim=1)
    
    def resnet_layer(self, num_residual_block):
        resnet = []

        inp_channels = out_channels = self.first_channels
        is_down = False
        for numBlock in num_residual_block:
            for _ in range(numBlock):
                resnet.append(Block(inp_channels, out_channels, is_down))
                inp_channels = out_channels
                is_down = False
            is_down = True
            out_channels = inp_channels * 2
        
        return nn.Sequential(*resnet), out_channels // 2
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.resnet(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        if self.vector:
            return x
        x = self.fc(x)
        # x = self.sm(x)
        return x
    
class Resnet_Encode_feature(Resnet):
    def __init__(self, 
                 single_vars,
                 press_vars,
                 press_levels,
                 encode_channels = 1,
                 **kwargs):
        super().__init__(**kwargs)
        self.single_vars = single_vars
        self.press_vars = press_vars
        self.press_levels = press_levels
        inp_channels = (single_vars + press_vars) * encode_channels
        self.conv1 = nn.Conv2d(inp_channels,
                               out_channels = self.first_channels,
                               kernel_size = 5,
                               stride = 2,
                               padding = 3,
                               bias = False)
        self.encode_list = []
        self.encode_list.extend([CNN2D_embed()] * single_vars)
        self.encode_list.extend([CNN2D_embed(press_levels=press_levels)] * press_vars)
        self.encode_list = nn.ModuleList(self.encode_list)
    
    def forward(self, x):
        
        print(x.shape)
        encode = []
        for i in range(self.single_vars):
            encode.append(self.encode_list[i](x[:, i:i+1]))
        
        for i in range(self.press_vars):
            encode.append(self.encode_list[self.single_vars + i](x[:, self.single_vars + i: self.single_vars + i + self.press_levels]))
        
        x = torch.cat(encode, dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.resnet(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        x = self.sm(x)
        return x



def create_model(out_dir = None,
                 model = Resnet,
                 **kwargs):
    
    if out_dir is not None:
        Path(os.path.join(out_dir, 'config')).mkdir(parents=True, exist_ok=True)
        save_kwargs = dict(locals())
        save_kwargs['model'] = model.__name__
        print(save_kwargs)
        json.dump(save_kwargs, open(os.path.join(out_dir, 'config', 'Model.json'), 'w'))
        print('[INFO]: Model config saved!')
        
    return model(**kwargs)

if __name__ == '__main__':
    model = create_model(model=Resnet,
                         inp_channels = 228,
                     num_residual_block = [2, 2, 2, 2], 
                     num_class = 2)#.to('cuda')
    
    # print(summary(model, 
    #               [(228, 33, 33)]))
    
    sample = torch.rand(64, 228, 33, 33)#.to('cuda')
    print(model(sample).shape)