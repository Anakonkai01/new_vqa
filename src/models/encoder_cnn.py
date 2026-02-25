"""
Docstring for models.encoder_cnn

- build a simple cnn: 
    - Conv2d 
    - BatchNorm2d
    - Rely
    - MaxPool2d
    - AdaptiveAvgPool2d: 
"""


import torch.nn as nn 
import torchvision.models as models

# helper function 
# note: W_out = W_in - kernel + 2 * padding
def conv_block(in_channels, out_channels):
    """  
    basic cnn block: conv -> bn -> relu -> maxpool 
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1), 
        nn.BatchNorm2d(out_channels), 
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2) # decrease spatial / 2 
    )



# SIMPLE CNN 
class SimpleCNN(nn.Module):
    
    def __init__(self, output_size=1024):
        super().__init__()

        self.features = nn.Sequential(
            conv_block(3, 64), 
            conv_block(64, 128),
            conv_block(128, 256), 
            conv_block(256, 512),
            conv_block(512, 1024),
        )
        
        # Squeeze spatial (7,7) -> (1,1)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # projection: 1024 -> output_size (match with hidden_size)
        self.fc = nn.Linear(1024, output_size)

    def forward(self, x):
        # x: (batch, 3, 224, 224)
        out = self.features(x) # -> (batch, 1024, 7, 7)
        out = self.pool(out) # -> (batch, 1024, 1, 1)
        out = out.flatten(1)  # -> (batch, 1024)
        out = self.fc(out)

        return out
    

    

# SIMPLE CNN — SPATIAL (used for Model C — Attention)
# Unlike SimpleCNN: does NOT mean-pool to (1,1)
# Keeps all spatial 7x7 = 49 regions so the decoder can attend over them
class SimpleCNNSpatial(nn.Module):
    def __init__(self, output_size=1024):
        super().__init__()

        # Same backbone as SimpleCNN — 5 conv_block layers
        self.features = nn.Sequential(
            conv_block(3, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512),
            conv_block(512, 1024),
        )

        # Project each region: 1024 -> output_size
        # Conv2d(kernel=1) is equivalent to a Linear applied independently to each spatial position
        self.proj = nn.Conv2d(1024, output_size, kernel_size=1)

    def forward(self, x):
        # x: (batch, 3, 224, 224)
        out = self.features(x)      # (batch, 1024, 7, 7)
        out = self.proj(out)        # (batch, output_size, 7, 7)

        # Flatten spatial: 7x7 = 49 regions
        # permute to move spatial dim last -> (batch, 49, output_size)
        batch = out.size(0)
        out = out.flatten(2)        # (batch, output_size, 49)
        out = out.permute(0, 2, 1)  # (batch, 49, output_size)

        return out  # (batch, 49, output_size) — 49 regional features


# Resnet101 for model C: without attention
class ResNetEncoder(nn.Module):
    def __init__(self, output_size=1024, freeze=True):
        super().__init__()
        
        # load resnet101 pretrain, default weight 
        resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        layers = list(resnet.children())[:-1] # remove the last layers (fc) 
        self.resnet = nn.Sequential(*layers)

        # freeze resnet 
        if freeze: 
            for param in self.resnet.parameters():
                param.requires_grad = False 
                
        # project to hidden size, output resnet (2048) -> ( hidden_size)
        self.fc = nn.Linear(2048, output_size)

    def forward(self, x):
        # x (batch, 3, 224, 224)
        out = self.resnet(x) # (batch, 2048, 1, 1)
        out = out.flatten(1) # (batch, 2048)
        out = self.fc(out) # (batch, hidden_size)

        return out # image feature 
        
        
    


    
# ResNet101 Spatial — used for Model D (Pretrained + Attention)
# Unlike ResNetEncoder: does NOT use final avgpool -> keeps spatial 7x7=49 regions
# ResNet101 structure: conv1->bn1->relu->maxpool -> layer1->2->3->4 -> avgpool -> fc
# [:-2] removes avgpool AND fc -> output is feature map (batch, 2048, 7, 7)
class ResNetSpatialEncoder(nn.Module):
    def __init__(self, output_size=1024, freeze=True):
        super().__init__()

        resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

        # Remove avgpool and fc (last 2 layers) -> keep spatial feature map
        layers = list(resnet.children())[:-2]  # output: (batch, 2048, 7, 7)
        self.resnet = nn.Sequential(*layers)

        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Project each region: 2048 -> output_size
        # Conv2d(kernel=1) = Linear applied independently over each of the 49 regions
        self.proj = nn.Conv2d(2048, output_size, kernel_size=1)

    def forward(self, x):
        # x: (batch, 3, 224, 224)
        out = self.resnet(x)        # (batch, 2048, 7, 7)
        out = self.proj(out)        # (batch, output_size, 7, 7)
        out = out.flatten(2)        # (batch, output_size, 49)
        out = out.permute(0, 2, 1)  # (batch, 49, output_size)
        return out                  # (batch, 49, output_size)


# test 
if __name__ == "__main__":
    import torch
    model = SimpleCNN(output_size=1024)
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    print(out.shape) # expect (4, 1024)

