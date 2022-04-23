from torch import nn


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv_layers = nn.Sequential(
         nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0),
         nn.BatchNorm2d(1),
         nn.ReLU(inplace=True),
         nn.MaxPool2d(kernel_size=2, stride=2),
         nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0),
         nn.BatchNorm2d(1),
         nn.ReLU(inplace=True),
         nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear = nn.Sequential(nn.Linear(40*40, 2))

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 40*40)
        out = self.linear(x)
        return out
