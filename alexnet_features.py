"""
AlexNet Features model
Just return features from alexnet
"""
import torch.nn as nn
from torchvision.models import alexnet
from torchvision.models import AlexNet_Weights

class AlexNetFeatures(nn.Module):
    def __init__(self):
        super(AlexNetFeatures, self).__init__()
        self.alexnet = alexnet(weights=AlexNet_Weights.DEFAULT)

    def forward(self, image):
        features = self.alexnet.features(image)
        return features