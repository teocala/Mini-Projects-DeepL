# Noise2Noise Model and its Implementation

This project is part of the Deep Learning course (EE-559) at EPFL. It is split into two parts:
- The first one aims at using PyTorch modules to implement a Neural Network suited for an image denoising task. 
- The second one aims at rebuilding from scratch said modules, without using PyTorch, in order to practice and demonstrate our knowledge of a Deep Learning framework.

## Repository Structure
This repository presents the following structure:
```
Proj_284260_342454_342763
├── Miniproject_1
│   ├── __init__.py
|   ├── main.py 
│   ├── model.py 
│   ├── bestmodel.pth
│   ├── Report_1.pdf
│   └── others
|       └── utilities.py
|
├── Miniproject_2
│   ├── __init__.py
|   ├── main.py 
│   ├── model.py 
│   ├── bestmodel.pth
│   ├── Report_2.pdf
│   └── others
|       ├── modules.py
|       └── extra
|           ├── pytorch_comparison.py
|           └── formulae
├── test_template.py
└── README.md
```

The file *test_template.py* has been provided by the teaching staff to get an idea of the automatic tests which will be conducted during evaluation.
## Miniproject 1 
The relevant pieces of code are in the *Miniproject_1/* folder.

The first part of this project consists in a PyTorch implementation of a *Noise2Noise* model. The file *model.py* contains our implementation of a *U-Net* applied to the image denoising problem. Within the module class, the class attribute **unet** can be changed to choose also another variant of the U-Net architecture, that is a Residual U-Net.
The file *main.py* can be run to start a training sequence for our model and it is possible to load our best performing model from the *bestmodel.pth* file. 

The *others/utilities.py* python file contains several functions and classes that we used to develop our model, such as the *PSNR* evaluation function and the two modules representing our networks.


The report for this part is available in the file *Report_1.pdf*.

## Miniproject 2
The relevant pieces of code are in the *Miniproject_2/* folder.

The second part of this project consists in the direct implementation of the modules used to construct a Neural Network. Namely, we implemented the following modules:
- 2D Convolutional Layer
- Nearest Neighbour Interpolation
- Upsampling (as a combination of Nearest Neighbour and Convolutional layers)
- Sigmoid activation function
- ReLU activation function
- SGD optimizer
- MSE loss function 
- Sequential container to wrap them all up in a model

Their implementation can be found in the *others/modules.py* file, and an example of a model built out of them is available in *model.py*. The file *main.py* can be run to start a training sequence for the example model. It is possible to save/load the trained model by calling the relative functions in the **Model()** class. An example is also shown in *main.py*. 

The *others/extra/* folder contains a LaTeX file reporting useful formulae for gradient computation in the Convolutional layer, as well as the python script we used to compare our implementation with PyTorch.

The report for this part is available in the file *Report_2.pdf*.

## Authors

* [Matteo Calafà](https://github.com/teocala)
* [Paolo Motta](https://github.com/paolomotta)
* [Thomas Rimbot](https://github.com/Thomas-debug-creator)
