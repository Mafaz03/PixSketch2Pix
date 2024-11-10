import torch
from torch import nn
import torchvision.models as models

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        self.features = ['0', '5', '10', '19', '20']
        self.model = models.vgg19(pretrained = True).features[:29]
    
    def forward(self, x):
        features = []
        
        for layer_idx, layer in enumerate(self.model):
            x = layer(x)

            if str(layer_idx) in self.features:
                features.append(x)
            
        return features
