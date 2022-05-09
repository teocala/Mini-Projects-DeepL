# -*- coding: utf-8 -*-

from torch import empty, cat, arange
from torch.nn.functional import fold, unfold
import modulus.py
# ATTENTION: DO NOT ADD ANY OTHER LIBRARY (see rules)

# torch.empty for an empty tensor
# torch.cat to concatenate more tensors
# torch.arange to create intervals
# fold/unfold to combine tensor blocks/batches (see end of projdescription)


# the code should work without autograd, don't touch it
import torch.set_grad_enabled
torch.set_grad_enabled(False)


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
        self.model = Sequential(
            Conv2d(input_channels = 3, output_channels = 32, kernel_size = 5, stride = 2),
            ReLU(),
            Conv2d(input_channels = 3, output_channels = 32, kernel_size = 5, stride = 2),
            ReLU(),
            NearestUpsampling(scale_factor = 2, input_channels = 3, output_channels = 32, kernel_size = 5, stride = 2), 
            ReLU(),
            NearestUpsampling(scale_factor = 2, input_channels = 3, output_channels = 32, kernel_size = 5, stride = 2), 
            Sigmoid()
        )
        self.loss = MSE()
        self.optimizer = SGD(self.parameters, lr=0.001)
        

    def load_pretrained_model ( self ) -> None :
        ## This loads the parameters saved in bestmodel .pth into the model
        pass
    

    def train ( self , train_input , train_target ) -> None :
        #: train˙input : tensor of size (N, C, H, W) containing a noisy version of the images.
        #: train˙target : tensor of size (N, C, H, W) containing another noisy version of the same images , which only differs from the input by their noise .
        batch_size = 100
        epochs = 5
        total_loss = 0

        for epoch in range(epochs):
            print(f'Epoch {epoch}/{epochs-1} Training Loss {total_loss}')
            total_loss = 0
            for batch_input, batch_target in zip(train_input.split(batch_size), train_target.split(batch_size)):
                output = self.predict(batch_input)
                loss = self.criterion(output, batch_target)
                total_loss += loss
                gradx = loss.backward() #loss w.r.t output of net
                gradx = self.model.backward(gradx) #loss w.r.t input of net
                self.optimizer.step()


    def predict ( self , test_input ) -> torch.Tensor :
        #: test˙input : tensor of size (N1 , C, H, W) that has to be denoised by the trained or the loaded network .
        #: returns a tensor of the size (N1 , C, H, W)
        y = self.model.forward(test_input)
        return y