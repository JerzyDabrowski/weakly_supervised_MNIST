The best test accuracy result of 97.2% was obtained using the ResNet18 model with the added 2 fully connected layers at the end. 
The activation function for each extra layer was relu.
The configurations tested were: 
- 1 additional fully connected layer at the end 0 -> test result accuracy ~ 94%
- ResNet18 with retrained last layer -> test result accuracy ~ 95.5%
- ResNet18 with all layers frozen except the last one zeroed -> test result accuracy ~ 93%
- convolutional network consisting of : -> test result accuracy ~ 70%
```
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
```

No augmentation techniques were used because the way the data were processed allowed the generation of varied set of different samples. 
Training without augmentation produced high results and the model itself did not overfit on the underlying data. 

The data were normalized. In order to support diffrent input sizes, padding with zeros was applied. 


To effectively train the ResNet model with modifications it is required to have more resources than when training the model presented above - than consist of few convolutional layers. 
Despite the varied data, the model generalizes well. 

Future improvements:
- Adding different techniques augmentation,
- Extending model architecture,
- Prunning





