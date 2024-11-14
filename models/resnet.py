'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from torchvision import models

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class DomainAdaptationLayer(nn.Module):
    def __init__(self, num_filters):
        super(DomainAdaptationLayer, self).__init__()
        # Learnable weights for each filter map
        self.adaptation_weights = nn.Parameter(torch.ones(num_filters))
    
    def forward(self, x):
        # Unsqueeze is used to broadcast the weights across spatial dimensions
        weighted_maps = x * self.adaptation_weights.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return weighted_maps

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, input_size=32, num_classes=10, resnet_model='resnet18', freeze_non_dal=False, freeze_dal=False, dal_on_main=False, unfreeze_linear=False, train_aux=False, norm_only=False, logger=None):
        super(ResNet, self).__init__()
        self.dal_on_main = dal_on_main
        self.freeze_non_dal = freeze_non_dal
        self.freeze_dal = freeze_dal
        self.logger = logger
        self.in_planes = 64

        # TODO: be able to load different types of resnet (18, 50, etc...)
        # Load a pretrained ResNet18 and modify for CIFAR-10
        # Define channel sizes based on ResNet architecture
        if resnet_model=="resnet50":
            self.channels = {
                'block1': 256,   # End of layer1
                'block2': 512,   # End of layer2
                'block3': 1024,  # End of layer3
                'block4': 2048   # End of layer4
            }
            resnet = models.resnet50(pretrained=True)
        else:  # resnet18
            self.channels = {
                'block1': 64,    # End of layer1
                'block2': 128,   # End of layer2
                'block3': 256,   # End of layer3
                'block4': 512    # End of layer4
            }
            resnet = models.resnet18(pretrained=True)
        if input_size==64:
            resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1, bias=False)
            self.pool_size = 2  # For Tiny-ImageNet
        else:
            resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            resnet.maxpool = nn.Identity()
            self.pool_size = 4
        
        self.block1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1
        )  # Output: 64 channels
        self.block2 = resnet.layer2  # Output: 128 channels
        self.block3 = resnet.layer3  # Output: 256 channels
        self.block4 = resnet.layer4  # Output: 512 channels
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.avgpool = resnet.avgpool
        
        # self.dal0 = DomainAdaptationLayer(filters[0])
        self.dal1 = DomainAdaptationLayer(self.channels['block1'])
        self.dal2 = DomainAdaptationLayer(self.channels['block2'])
        self.dal3 = DomainAdaptationLayer(self.channels['block3'])
        self.dal4 = DomainAdaptationLayer(self.channels['block4'])
        
        if freeze_non_dal:
            self._freeze_non_dal_modules()
            
        if freeze_dal:
            self._freeze_dal_modules()

        if norm_only:
            self._freeze_dal_modules()
            self._freeze_non_dal_modules()
            self.unfreeze_all_norm_params()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def set_return_features(self, value: bool):
        self.return_features = value

    def is_norm_layer(self, param_name: str) -> bool:
        """Check if parameter belongs to any normalization layer."""
        norm_types = ['bn', 'gn', 'ln']  # batch norm, group norm, layer norm
        return any(norm_type in param_name for norm_type in norm_types) and ('weight' in param_name or 'bias' in param_name)

    def unfreeze_all_norm_params(self):
        """Unfreeze all normalization parameters (γ, β) in the model."""
        for name, param in self.named_parameters():
            if self.is_norm_layer(name):
                param.requires_grad = True
        
    def freeze_all_except_layer(self, layer_id: int, bn_only: bool):
        # if self.logger:
        #     self.logger.info(f"Freezing all layers except layer {layer_id}")
        
        # Loop through all named parameters
        for name, param in self.named_parameters():
            if f'layer{layer_id}' not in name:
                # Freeze all layers not part of the target layer
                param.requires_grad = False
            else:
                if bn_only:
                    # If bn_only is True, only unfreeze BatchNorm affine parameters
                    if 'bn' in name and ('weight' in name or 'bias' in name):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    # If bn_only is False, unfreeze all parameters in the target layer
                    param.requires_grad = True
    
    def freeze_fc_layers(self):
        if self.logger:
            self.logger.info("Freezing fully connected weights")
        for name, param in self.named_parameters():
            if 'linear' in name:
                param.requires_grad = False

    def _freeze_non_dal_modules(self):
        if self.logger:
            self.logger.info("Freezing non-attention weights")
        for name, param in self.named_parameters():
            if 'dal' not in name:
                param.requires_grad = False
                
    def _unfreeze_non_dal_modules(self):
        if self.logger:
            self.logger.info("Unfreezing non-attention weights")
        for name, param in self.named_parameters():
            if 'dal' not in name:
                param.requires_grad = True
    
    def _freeze_dal_modules(self):
        if self.logger:
            self.logger.info("Freezing attention weights")
        for name, param in self.named_parameters():
            if 'dal' in name:
                param.requires_grad = False
                
    def _unfreeze_dal_modules(self):
        if self.logger:
            self.logger.info("Unfreezing attention weights")
        for name, param in self.named_parameters():
            if 'dal' in name:
                param.requires_grad = True

    def forward(self, x):
        
        dal_feature_maps = {}
        
        main_out = self.block1(x) 
        main_out = self.dal1(main_out)
        dal_feature_maps[1] = main_out
        
        main_out = self.block2(main_out)
        main_out = self.dal2(main_out)
        dal_feature_maps[2] = main_out
        
        main_out = self.block3(main_out)
        main_out = self.dal3(main_out)
        dal_feature_maps[3] = main_out
        
        main_out = self.block4(main_out)
        main_out = self.dal4(main_out)
        dal_feature_maps[4] = main_out
        
        main_out = self.avgpool(main_out)
        main_out = main_out.view(main_out.size(0), -1)
        main_out = self.linear(main_out)
        
        return main_out


def ResNet18(num_classes, freeze_non_dal=False, freeze_dal=False, dal_on_main=False, logger=None):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, freeze_non_dal=freeze_non_dal, freeze_dal=freeze_dal, dal_on_main=dal_on_main, logger=logger)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50(num_classes, freeze_non_dal=False, freeze_dal=False, dal_on_main=False, logger=None):
    return ResNet(Bottleneck, [3, 4, 6, 3], resnet_model='resnet50', num_classes=num_classes, freeze_non_dal=freeze_non_dal, freeze_dal=freeze_dal, dal_on_main=dal_on_main, logger=logger)


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])

