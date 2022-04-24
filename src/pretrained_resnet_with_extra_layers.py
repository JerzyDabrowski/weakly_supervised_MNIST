import torchvision.models as models
from torch import nn


NUM_CLASSES = 2
resnet = models.resnet18(pretrained=True)
#
# resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
last_layer = resnet.fc.in_features
resnet.fc = nn.Linear(last_layer, 256)

# Added fully connected layers on the end


class FcNet(nn.Module):
    def __init__(self):
        super(FcNet, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, NUM_CLASSES),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.fc_layers(x)

        return out


end_of_net = FcNet()
resnet_and_fc = nn.Sequential(resnet, end_of_net)
