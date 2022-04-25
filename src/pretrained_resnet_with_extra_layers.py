import torchvision.models as models
from torch import nn


NUM_CLASSES = 2

resnet = models.resnet18(pretrained=True)
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


# combine models into one
end_of_net = FcNet()
resnet_and_fc = nn.Sequential(resnet, end_of_net)
