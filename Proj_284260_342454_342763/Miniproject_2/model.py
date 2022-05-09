# -*- coding: utf-8 -*-

from torch import empty, cat, arange
from torch.nn.functional import fold, unfold
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


# ABSTRACT MODULE CLASS (required)
class Module (object):
    def forward (self, *input):
        raise NotImplementedError
    def backward (self , *gradwrtoutput):
        raise NotImplementedError
    def param (self):
        return []



# IMPLEMENTATION OF MODULES:
    
    
class Conv2d(Module):
    
    def __init__(self, input_channels, output_channels, kernel_size, stride) -> None:
        super().__init__()
        
        self.x = empty()
        self.gradx = empty()
        self.input = empty()
        self.gradoutput = empty()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        # default: padding = 0, dilation=1
        
        # from https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        self.weight = empty((self.output_channels, self.input_channels, self.kernel_size, self.kernel_size))
        self.bias = empty((self.output_channels))
        
    
    def compute_output_shape(self, *input): # for the time being, not used
        H = (input.shape[2] - (self.kernel_size-1)-1)/2 + 1
        W = (input.shape[3] - (self.kernel_size-1)-1)/2 + 1
        return [H.floor(), W.floor()]
    
    def forward(self, *input): # see end of projdescription
        unfolded = unfold(input, self.kernel_size)
        self.input = input
        self.x = self.weight @ unfolded + self.bias 
        return self.x   
        
    def backward(self, *gradwrtoutput):
        gradux =  (self.weight).T.dot(gradwrtoutput) #derivative w.r.t. unfold(x)
        self.gradoutput = gradwrtoutput
        self.gradx = fold(gradux, input.shape[2:4],self.kernel_size) #derivative w.r.t. x
        # fold is not exactly the inverse of unfold but does exactly the weight sharing for the computation of the gradient
        return self.gradx

    def param(self):
        dldw = self.gradoutput.dot(self.input.T)
        dldb = self.gradoutput
        
        # list of zip to create list of tuples
        return list(zip(cat(self.weight,self.bias), cat(dldw, dldb)))
    




class NearestNeighbor(Module):  
    def __init__(self, scale_factor) -> None:
        super().__init__()
        
        self.x = empty()
        self.gradx = empty()
        self.input = empty()
        
        # NN
        self.scale_factor = scale_factor
        self.NN_output_shape = []


    def forward (self, *input):
        self.input = input
        # Compute the NN output shape from the input size and the scale factor
        self.NN_output_shape = [input.shape[0], input.shape[1], [self.scale_factor * dim for dim in input.shape[2:]]]
        NN_interp = empty(self.NN_output_shape)

        # Apply NN interpolation
        for i in range(self.scale_factor):
            for j in range(self.scale_factor):
                NN_interp[:,:,i::self.scale_factor,j::self.scale_factor] = input
        
        return self.NN_interp


    def backward (self , *gradwrtoutput):
        # Get gradient of convolution 
        self.gradoutput = gradwrtoutput

        # Since we used NN interpolation, we have to sum up the derivatives
        # in the gradient of the convolution on each block
        grad = empty(self.input.shape)
        for i in range(input.shape[2]):
            for j in range(input.shape[3]):
                i_output = i * self.scale_factor
                j_output = j * self.scale_factor
                grad[:,:,i,j] = gradwrtoutput[:,:,i_output:i_output+self.scale_factor,j_output:j_output+self.scale_factor].sum()

        self.gradx = grad
        return self.gradx
    
    
    
    
class ReLU(Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.input = empty()
        self.output = empty()
        self.gradx = empty()
    
    def forward(self, *input):
        output = empty(input.shape)
        output[input>0] = input
        self.input = input
        self.output = output
        return output
        
    def backward(self, *gradwrtoutput):
        self.gradx =  gradwrtoutput.multiply(self.sigmaprime(self.input))
        return self.gradx
        
    def sigmaprime(*input):
        output = empty(input.shape)
        output.fill_(0.0)
        output[input>0] = 1.0
        return output
     
        
     
    
    
class Sigmoid(Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.input = empty()
        self.output = empty()
        self.gradx = empty()
    
    def forward(self, *input):
        self.input = input
        self.output = self.sigma(input)
        return self.output
        
    def backward(self, *gradwrtoutput):
        self.gradx =  gradwrtoutput.multiply(self.sigmaprime(self.input))
        return self.gradx
    
    def sigma(self, *input):
        return input.exp()/(1+ input.exp())
    
    def sigmaprime(self, *input):
        return self.sigma(input)*(1-self.sigma(input))
    
    
    
class MSE(Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.input = empty()
        self.target = empty()
        self.gradx = empty()
    
    def forward(self, *input, target):
        self.input = input
        self.target = target
        return pow(input-target,2).sum()
        
    def backward(self):
        self.gradx = 2*(self.input-self.target)
        return self.gradx
    
    
    
    
    
class SGD(Module):
    def __init__(self, *parameters, lr) -> None:
        super().__init__()
        self.parameters = parameters
        self.lr = lr
    
    def step(self, *parameters):
        idx = torch.empty(1)
        idx.random_(0,len(parameters))
        parameters = parameters - self.lr * parameters[idx.item()][1]
        



class Sequential(Module):
    
    def __init__(self, *args) -> None:
        super().__init__()
        self.args = args
        self.input = empty()
        self.output = empty()
        self.gradx = empty()
    
    def forward(self, *input):
        self.input = input
        output = input.copy()
        for module in self.args:
            output = module.forward(output)
        self.output = output
        return output
        
    def backward(self, *gradwrtoutput):
        gradx = gradwrtoutput.copy()
        for module in self.args:
            gradx = module.backward(gradx)
        self.gradx = gradx
        return gradx
    
    def param(self):
        param = []
        for module in self.args:
            param = param.append(module.param())
        return param
    
    
    
    
class NearestUpsampling(Sequential):  
    def __init__(self, scale_factor, input_channels, output_channels, kernel_size, stride) -> None:
        super().__init__(NearestNeighbor(scale_factor), Conv2d(input_channels, output_channels, kernel_size, stride))
        
        
        