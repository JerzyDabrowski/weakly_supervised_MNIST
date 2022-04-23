import torchvision. models as models
from torch import nn


NUM_CLASSED = 2
resnet = models.resnet18(pretrained=True)
resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
last_layer = resnet.fc.in_features
resnet.fc = nn.Linear(last_layer, NUM_CLASSED)


#print(resnet)
