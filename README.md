# weakly_supervised_MNIST

The project presents training, testing and visualization for a selected ResNet Model extended with additional end layers. 

Model parameters are saved when training is completed. 

The visualizations and the learning process are saved in tensorboard.


Executing the command `tensorboard --logdir=tensorboard_res` will enable the collection of the information listed above.

The model can be found in file `pretrained_resnet_with_extra_layers.py`
In order to analyze the layers, it is recommended to use the `GRAPHS` tab in tensorboard


Running the `resnet_training.py` file starts the learning process 

