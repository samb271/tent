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
    def __init__(self, num_filters, threshold=1e-3):
        super(DomainAdaptationLayer, self).__init__()
        # Learnable weights for each filter map
        self.adaptation_weights = nn.Parameter(torch.ones(num_filters))
        self.threshold = threshold  # Threshold for deactivating weights
    
    def forward(self, x):
        # Create a mask where weights below the threshold are set to 0
        # mask = (self.adaptation_weights.abs() >= self.threshold).float()
        # # Apply the mask to the adaptation weights
        # masked_weights = self.adaptation_weights * mask
        # Unsqueeze is used to broadcast the weights across spatial dimensions
        weighted_maps = x * self.adaptation_weights.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return weighted_maps
    
    def get_sparse_indices(self):
        """Returns indices of non-zero weights"""
        return torch.nonzero(self.adaptation_weights.abs() >= self.threshold).squeeze()
    
    def get_active_features_count(self):
        """Returns the number of non-zero weights"""
        return (self.adaptation_weights.abs() >= self.threshold).sum().item()
    
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )
        
    def forward(self, x):
        return self.net(x)

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

class ResNetReconstruct(nn.Module):
    def __init__(self, block, num_blocks, input_size=32, num_classes=10, resnet='resnet18', logger=None):
        super(ResNetReconstruct, self).__init__()
        self.logger = logger
        self.in_planes = 64

        # TODO: be able to load different types of resnet (18, 50, etc...)
        # Load a pretrained ResNet18 and modify for CIFAR-10
        # Define channel sizes based on ResNet architecture
        if resnet=="resnet50":
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

        dim = self.channels['block4']
        # Projection layer with ReLU activation
        self.projection = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU()
        )

        # Initialize all linear layers as identity matrices
        def init_as_identity(m):
            if isinstance(m, nn.Linear):
                # Initialize weights as identity matrix
                m.weight.data = torch.eye(m.weight.data.size(0))
                if m.bias is not None:
                    # Initialize bias as zeros
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                # Initialize BatchNorm layers with identity transformation
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
                m.running_mean.zero_()
                m.running_var.fill_(1.0)
        
        # Apply identity initialization to all layers in projection
        self.projection.apply(init_as_identity)
        
        # self.dal0 = DomainAdaptationLayer(filters[0])
        self.dal1 = DomainAdaptationLayer(self.channels['block1'])
        self.dal2 = DomainAdaptationLayer(self.channels['block2'])
        self.dal3 = DomainAdaptationLayer(self.channels['block3'])
        self.dal4 = DomainAdaptationLayer(self.channels['block4'])

        self.freeze_all_except_layer("projection", False, with_norm=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
        
    def is_norm_layer(self, param_name: str) -> bool:
        """Check if parameter belongs to any normalization layer."""
        norm_types = ['bn', 'gn', 'ln']  # batch norm, group norm, layer norm
        return any(norm_type in param_name for norm_type in norm_types) and ('weight' in param_name or 'bias' in param_name)

    def unfreeze_all_norm_params(self):
        """Unfreeze all normalization parameters (γ, β) in the model."""
        for name, param in self.named_parameters():
            if self.is_norm_layer(name):
                param.requires_grad = True

    def freeze_all_except_layer(self, layer_id: str, norm_only: bool, with_norm: bool):
        """Freeze all layers except specified layer, with options for normalization parameters."""
        # First freeze everything
        for name, param in self.named_parameters():
            param.requires_grad = False
        
        # If with_norm is True, unfreeze all normalization parameters first
        if with_norm:
            self.unfreeze_all_norm_params()
        
        # Then handle the target layer
        for name, param in self.named_parameters():
            if f'{layer_id}' in name:
                if norm_only:
                    # Only unfreeze normalization parameters in target layer
                    if self.is_norm_layer(name):
                        param.requires_grad = True
                else:
                    # Unfreeze all parameters in target layer
                    param.requires_grad = True

    def forward(self, x):
                
        # ~~ LAYER 1 ~~
        # Extract features
        main_out = self.block1(x) 
        dal_out = self.dal1(main_out)
        main_out = self.block2(dal_out)
        dal_out = self.dal2(main_out)
        main_out = self.block3(dal_out)
        dal_out = self.dal3(main_out)
        main_out = self.block4(dal_out)
        dal_out = self.dal4(main_out)
        
        #~~~ MAIN TASK ~~~

        main_out = self.avgpool(dal_out)
        main_out = main_out.view(main_out.size(0), -1)
        projected = self.projection(main_out)

        main_out = self.linear(projected)
        
        return main_out


def ResNet18Reconstruct(logger=None):
    return ResNetReconstruct(BasicBlock, [2, 2, 2, 2], logger=logger)


def ResNet34():
    return ResNetReconstruct(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNetReconstruct(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNetReconstruct(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNetReconstruct(Bottleneck, [3, 8, 36, 3])

