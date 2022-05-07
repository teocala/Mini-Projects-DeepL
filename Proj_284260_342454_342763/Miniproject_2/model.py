# -*- coding: utf-8 -*-

from torch import empty, cat, arange
from torch.nn.functional import fold, unfold
# ATTENTION: DO NOT ADD ANY OTHER LIBRARY (see rules)

# torch.empty for an empty tensor
# torch.cat to concatenate more tensors
# torch.arange to create intervals
# fold/unfold to combine tensor blocks/batches (see end of projdescription)


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


# ABSTRACT MODULE CLASSE (required)
class Module (object):
    def forward (self, *input):
        raise NotImplementedError
    def backward (self , *gradwrtoutput):
        raise NotImplementedError
    def param (self):
        return []



# IMPLEMENTATION OF MODULES:
    

#class Conv1(Module):
#class ReLU1(Module):
    
class Conv2(Module):
    
    def __init__(self) -> None:
        super().__init__()
        
        self.x = empty()
        self.gradx = empty()
        self.input = empty()
        self.gradoutput = empty()
        
        self.input_channels = 64 #parametri a caso, da fixare dopo
        self.output_channels = 64
        self.kernel_size = 5
        self.stride = 2
        # default: padding = 0, dilation=1
        
        # from https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        self.weight = empty((self.output_channels, self.input_channels, self.kernel_size, self.kernel_size))
        self.bias = empty((self.output_channels))
        
    
    def compute_output_shape(self, *input):
        H = (input.shape[2] - (self.kernel_size-1)-1)/2 + 1
        W = (input.shape[3] - (self.kernel_size-1)-1)/2 + 1
        return [H.floor(), W.floor()]
    
    def inverse_unfold(self, *input): 
        #from https://pytorch.org/docs/stable/generated/torch.nn.Fold.html
        input_ones = torch.empty(input.shape, dtype=input.dtype)
        input_ones.fill_(1.0)
        divisor = fold(unfold(input_ones))
        return divisor.inverse()@fold(input)
    
    def forward(self, *input): # see end of projdescription
        unfolded = unfold(input, self.kernel_size)
        self.input = input
        self.x = self.weight @ unfolded + self.bias 
        
    def backward(self, *gradwrtoutput):
        dldux =  (self.weight).T.dot(gradwrtoutput) #derivative w.r.t. unfold(x)
        self.gradoutput = gradwrtoutput
        self.gradx = self.inverse_unfold(dldux) #derivative w.r.t. x
    
    def param(self):
        dldw = self.gradoutput.dot(self.input.T)
        dldb = self.gradoutput
        return [cat(self.weight,self.bias), cat(dldw, dldb)]
    

#class ReLU2(Module):
#class Upsampling1(Module):  

    
class ReLU3(Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.x_input = empty()
        self.x_output = empty()
        self.gradx = empty()
    
    def forward(self, *input):
        output = empty(input.shape)
        output[input>0] = input
        self.x_input = input
        self.x_output = output
        return output
        
    def backward(self, *gradwrtoutput):
        self.gradx =  gradwrtoutput.multiply(self.sigmaprime(self.x))
        return self.gradx
        
    def sigmaprime(*input):
        output = empty(input.shape)
        output.fill_(0.0)
        output[input>0] = 1.0
        return output
    
    
#class Upsampling2(Module):   
    
    
class Sigmoid(Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.x_input = empty()
        self.x_output = empty()
        self.gradx = empty()
    
    def forward(self, *input):
        self.x_input = input
        self.x_output = self.sigma(input)
        return self.x_output
        
    def backward(self, *gradwrtoutput):
        self.gradx =  gradwrtoutput.multiply(self.sigmaprime(self.x))
        return self.gradx
    
    def sigma(self, *input):
        return input.exp()/(1+ input.exp())
    
    def sigmaprime(self, *input):
        return self.sigma(input)*(1-self.sigma(input))
        