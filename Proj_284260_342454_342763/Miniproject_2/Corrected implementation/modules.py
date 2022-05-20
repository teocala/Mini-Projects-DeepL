# -*- coding: utf-8 -*-

from torch import empty, cat, arange, Tensor
from torch.nn.functional import fold, unfold
# ATTENTION: DO NOT ADD ANY OTHER LIBRARY (see rules)

# torch.empty for an empty tensor
# torch.cat to concatenate more tensors
# torch.arange to create intervals
# fold/unfold to combine tensor blocks/batches (see end of projdescription)

import torch #TO REMOVE

# the code should work without autograd, don't touch it
from torch import set_grad_enabled
set_grad_enabled(False)


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
    def update_params(self, new_params):
        pass



# IMPLEMENTATION OF MODULES:
    
    
class Conv2d(Module):
    
    def __init__(self, input_channels, output_channels, kernel_size, stride) -> None:
        super().__init__()
        
        self.x = Tensor()
        self.x_shape = ()
        self.gradx = Tensor()
        self.input = Tensor()
        self.gradoutput = Tensor()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        # default: padding = 0, dilation=1
        
        # from https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        self.weight = empty((self.output_channels, self.input_channels, self.kernel_size[0], self.kernel_size[1]))
        self.bias = empty((self.output_channels))
        self.dldw = empty((self.output_channels, self.input_channels, self.kernel_size[0], self.kernel_size[1]))
        self.dldb = empty((self.output_channels))


        # Random initialization 
        N = self.output_channels * self.input_channels * self.kernel_size[0] * self.kernel_size[1]
        self.weight.uniform_(-1/(N**0.5), 1/(N**0.5))
        self.bias.uniform_(-1/(N**0.5), 1/(N**0.5))

        
    
    def compute_output_shape(self, *input): # for the time being, not used
        H = (input[0].shape[2] - (self.kernel_size[0]-1)-1)/self.stride + 1
        W = (input[0].shape[3] - (self.kernel_size[1]-1)-1)/self.stride + 1
        return [int(H), int(W)]
    
    def forward(self, *input): # see end of projdescription
        unfolded = unfold(input[0], kernel_size=self.kernel_size, stride=self.stride)
        self.input = input[0]
        self.x = self.weight.view(self.output_channels, -1) @ unfolded + self.bias.view(1,-1,1)

        H, W = self.compute_output_shape(input[0])

        self.x_shape = self.x.shape
        
        self.x = self.x.view(self.input.shape[0], self.output_channels, H, W)
        return self.x   
        
    def backward(self, *gradwrtoutput):
        self.gradoutput = gradwrtoutput[0]
        G = self.gradoutput.view(self.x_shape).transpose_(1,2)
        W = self.weight.view(self.output_channels, -1)
        gradux = torch.tensordot(G,W, dims=1)
        gradux = gradux.transpose_(1,2)
        
        

        folded = fold(gradux, self.input.shape[2:], kernel_size=self.kernel_size, stride=self.stride) #derivative w.r.t. x
        # fold is not exactly the inverse of unfold but does exactly the weight sharing for the computation of the gradient
        self.gradx = folded
        return self.gradx

    def param(self):
        if not len(self.gradoutput) == 0: 
            
            unfolded = unfold(self.input, kernel_size=self.kernel_size, stride=self.stride)
            G = self.gradoutput.view(self.x_shape).transpose_(0,1)
            U = unfolded.transpose_(1,2)
            
            self.dldw = torch.tensordot(G,U,dims=2)
            self.dldw = self.dldw.view(self.weight.shape)
            self.dldb = (self.gradoutput.view(self.x_shape).sum(2)).sum(0) # again weight sharing
            self.dldb = self.dldb.view(self.bias.shape)

        return [[self.weight, self.dldw],[self.bias, self.dldb]]

    def update_params(self, new_params):
        self.weight = new_params[0]
        self.bias = new_params[1]
    




class NearestNeighbor(Module):  
    def __init__(self, scale_factor) -> None:
        super().__init__()
        
        self.x = Tensor()
        self.NN_interp = Tensor()
        self.gradx = Tensor()
        self.input = Tensor()
        
        # NN
        self.scale_factor = scale_factor
        self.NN_output_shape = []


    def forward (self, *input):
        self.input = input[0]
        # Compute the NN output shape from the input size and the scale factor
        self.NN_output_shape = [self.input.shape[0], self.input.shape[1]] + [self.scale_factor * dim for dim in self.input.shape[2:]]
        self.NN_interp = empty(self.NN_output_shape)

        # Apply NN interpolation
        for i in range(self.scale_factor):
            for j in range(self.scale_factor):
                self.NN_interp[:,:,i::self.scale_factor,j::self.scale_factor] = input[0]
        
        return self.NN_interp


    def backward (self , *gradwrtoutput):
        # Since we used NN interpolation, we have to sum up the derivatives
        # in the gradient of the convolution on each block
        grad = empty(self.input.shape)
        for i in range(self.input.shape[2]):
            for j in range(self.input.shape[3]):
                i_output = i * self.scale_factor
                j_output = j * self.scale_factor
                grad[:,:,i,j] = gradwrtoutput[0][:,:,i_output:i_output+self.scale_factor,j_output:j_output+self.scale_factor].sum()

        self.gradx = grad
        return self.gradx
    
    
    
    
class ReLU(Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.input = Tensor()
        self.output = Tensor()
        self.gradx = Tensor()
    
    def forward(self, *input):
        self.input = input[0]
        output = empty(input[0].shape)
        output = self.input.where(self.input > 0, 0*self.input)
        self.output = output
        return output
        
    def backward(self, *gradwrtoutput):
        self.gradx =  gradwrtoutput[0].multiply(self.sigmaprime(self.input))
        return self.gradx
        
    def sigmaprime(self, *input):
        output = empty(input[0].shape)
        output.fill_(0.0)
        id = empty(input[0].shape)
        id.fill_(1.0)
        output = id.where(input[0] > 0, output)
        return output
        
     
    
    
class Sigmoid(Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.input = Tensor()
        self.output = Tensor()
        self.gradx = Tensor()
    
    def forward(self, *input):
        self.input = input[0]
        self.output = self.sigma(input[0])
        return self.output
        
    def backward(self, *gradwrtoutput):
        self.gradx =  gradwrtoutput[0].multiply(self.sigmaprime(self.input))
        return self.gradx
    
    def sigma(self, *input):
        return input[0].exp()/(1+ input[0].exp())
    
    def sigmaprime(self, *input):
        return self.sigma(input[0])*(1-self.sigma(input[0]))
    
    
    
class MSE(Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.x = Tensor()
        self.target = Tensor()
        self.gradx = Tensor()
    
    def forward(self, *input):
        self.target = input[1]
        self.x = input[0]
        return pow(input[0]-input[1],2).sum() / self.x.shape[0]
        
    def backward(self):
        self.gradx = 2/self.x.shape[0] * (self.x-self.target)
        return self.gradx
    
    
    
    
    
class SGD(Module):
    def __init__(self, *parameters, lr) -> None:
        super().__init__()
        self.parameters = parameters[0]
        self.lr = lr
    
    def step(self, model):
        for module in model.args:
            rhs = []
            for p in module.param():
                a = torch.empty(p[0].shape)
                a.copy_(p[0])
                rhs.append(a - self.lr*p[1])
                print (torch.mean(abs(a)))
                # print(p[1])
                # print(f"p: {p[0]}", f"grad_p : {p[1]}")

            module.update_params(rhs)
        # for p in self.parameters:
            # a = torch.empty(p[0].shape)
            # a.copy_(p[0])
            # p[0] = p[0] - self.lr * p[1]
            # print(torch.count_nonzero(a-p[0]))
        # wrong formula, need correction
        



class Sequential(Module):
    
    def __init__(self, *args) -> None:
        super().__init__()
        self.args = args
        self.input = Tensor()
        self.output = Tensor()
        self.gradx = Tensor()

    
    def forward(self, *input):
        self.input = input[0]
        self.output = empty(self.input.shape)
        self.output.copy_(self.input)
        for module in self.args:
            self.output = module.forward(self.output)
        return self.output
        
    def backward(self, *gradwrtoutput):
        self.gradx = empty(gradwrtoutput[0].shape)
        self.gradx.copy_(gradwrtoutput[0])
        for module in self.args[::-1]:
            self.gradx = module.backward(self.gradx)
        return self.gradx

    def param(self):
        param = [module.param() for module in self.args][0]
        # for module in self.args:
        #     param = param + module.param()
        return param
    
    
    
    
class NearestUpsampling(Sequential):  
    def __init__(self, scale_factor, input_channels, output_channels, kernel_size, stride) -> None:
        super().__init__(NearestNeighbor(scale_factor), Conv2d(input_channels, output_channels, kernel_size, stride))
        
    def param(self): # Return parameters of Convolutional layer
        return self.args[1].param()

    def update_params(self, new_params): # Update parameters of Convolutional layer 
        self.args[1].update_params(new_params)
        