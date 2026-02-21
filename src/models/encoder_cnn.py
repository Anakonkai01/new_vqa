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
    

    

# Resnet101 
class ResNetEncoder(nn.Module):
    def __init__(self, output_size=1024, freeze=True):
        super().__init__()
        
        # load resnet101 pretrain, default weight 
        resnet = models.resnet101(weights=models.ResNet101_Weights("Default"))
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
        
        
    


    
# test 
if __name__ == "__main__":
    import torch
    model = SimpleCNN(output_size=1024)
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    print(out.shape) # expect (4, 1024)

