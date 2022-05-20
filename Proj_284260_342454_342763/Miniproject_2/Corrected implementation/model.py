# -*- coding: utf-8 -*-

from torch import empty, cat, arange, Tensor
from torch.nn.functional import fold, unfold
from modules import *
# ATTENTION: DO NOT ADD ANY OTHER LIBRARY (see rules)

# torch.empty for an empty tensor
# torch.cat to concatenate more tensors
# torch.arange to create intervals
# fold/unfold to combine tensor blocks/batches (see end of projdescription)


# the code should work without autograd, don't touch it
from torch import set_grad_enabled
set_grad_enabled(False)

import torch # TO REMOVE

# REFERENCE STRUCTURE:
"""
Sequential (Conv (stride 2),
            ReLU,
            Conv (stride 2),
            ReLU,
            Upsampling,
            ReLU,
            Upsampling,
            Sigmoid)
"""



### For mini - project 2
class Model () :
    def __init__ ( self ) -> None :
        ## instantiate model + optimizer + loss function + any other stuff you need
        self.model = Sequential(# N, 3, 32, 32
            Conv2d(input_channels = 3, output_channels = 32, kernel_size = (5,5), stride = 2), # N, 32, 14, 14
            ReLU(),
            Conv2d(input_channels = 32, output_channels = 32, kernel_size = (3,3), stride = 2), # N, 32, 5, 5
            ReLU(),
            NearestUpsampling(scale_factor = 5, input_channels = 32, output_channels = 32, kernel_size = (5,5), stride = 2), #  N, 32, 15, 15
            ReLU(),
            NearestUpsampling(scale_factor = 5, input_channels = 32, output_channels = 3, kernel_size = (3,3), stride = 2), # N, 3, 32, 32
            Sigmoid()
            # Conv2d(input_channels = 3, output_channels = 32, kernel_size = (3,3), stride = 2), # N, 32, 15, 15
            # ReLU()
        )
        self.criterion = MSE()
        self.optimizer = SGD(self.model.param(), lr=0.00001)

    def load_pretrained_model ( self ) -> None :
        ## This loads the parameters saved in bestmodel .pth into the model
        pass
    

    def train ( self , train_input , train_target ) -> None :
        #: train˙input : tensor of size (N, C, H, W) containing a noisy version of the images.
        #: train˙target : tensor of size (N, C, H, W) containing another noisy version of the same images , which only differs from the input by their noise .
        set_grad_enabled(False)
        batch_size = 500
        epochs = 10

        # train_target_test = torch.randn((500,32,15,15)) # Just to work with the right dimension with one Conv2D

        for epoch in range(epochs):
            total_loss = 0
            for batch_input, batch_target in zip(train_input.split(batch_size), train_target.split(batch_size)):
                output = self.predict(batch_input)
                loss = self.criterion.forward(output, batch_target)
                total_loss += loss
                gradx = self.criterion.backward() #loss w.r.t output of net
                self.gradx = self.model.backward(gradx) #loss w.r.t input of net
                self.optimizer.step(self.model)
            # for p in self.model.param():
            #     print(f"p: {p[0]}", f"grad_p : {p[1]}")
            print(f'Epoch {epoch}/{epochs-1} Training Loss {total_loss}')


    def predict ( self , test_input ) -> Tensor:
        #: test˙input : tensor of size (N1 , C, H, W) that has to be denoised by the trained or the loaded network .
        #: returns a tensor of the size (N1 , C, H, W)
        y = self.model.forward(test_input)
        return y