# -*- coding: utf-8 -*-

from torch import empty, cat, arange, Tensor
from torch.nn.functional import fold, unfold
# ATTENTION: DO NOT ADD ANY OTHER LIBRARY (see rules)

# torch.empty for an empty tensor
# torch.cat to concatenate more tensors
# torch.arange to create intervals
# fold/unfold to combine tensor blocks/batches (see end of projdescription)

from torch import stack # temporary


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
    def gradparam (self, idx):
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
        self.stoch_x_shape = [1, self.x.shape[1], self.x_shape[2]]
        self.x = self.x.view(self.input.shape[0], self.output_channels, H, W)
        return self.x   
        
    def backward(self, *gradwrtoutput):
        gradux =  (self.weight.view(self.output_channels, -1)).T @ gradwrtoutput[0].view(self.x_shape) #derivative w.r.t. unfold(x)
        self.gradoutput = gradwrtoutput[0]

        folded = fold(gradux, self.input.shape[2:], kernel_size=self.kernel_size, stride=self.stride) #derivative w.r.t. x
        # fold is not exactly the inverse of unfold but does exactly the weight sharing for the computation of the gradient
        self.gradx = folded
        return self.gradx

    def param(self):
        res = [self.weight, self.bias]
        return res
    
    def gradparam(self, idx):
        if not len(self.gradoutput) == 0: 
            # we need the idx term and not the 100 batch elements
            unfolded = unfold(self.input[idx:idx+1,:,:,:], kernel_size=self.kernel_size, stride=self.stride)
            self.dldw = self.gradoutput[idx:idx+1,:,:,:].view(self.stoch_x_shape) @ unfolded.transpose(1,2)
            self.dldw = self.dldw.view(self.weight.shape)
            
            self.dldb = self.gradoutput[idx:idx+1,:,:,:].view(self.stoch_x_shape).sum(2) # again weight sharing
            self.dldb = self.dldb.view(self.bias.shape)
        return [self.dldw, self.dldb]
    
    
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
        return pow(input[0]-input[1],2).sum()
        
    def backward(self):
        self.gradx = 2*(self.x-self.target)
        return self.gradx
    
    def stoch_backward(self, idx):
        self.gradx = 2*(self.x[idx] - self.target[idx])
        return self.gradx
        
    
    
    
    
    
class SGD(Module):
    def __init__(self, lr) -> None:
        super().__init__()
        self.lr = lr
    
    def step(self, model,idx):
        for module in model.args:
            params = module.param()
            stochgradparams = module.gradparam(idx)
            rhs = []
            for i in range(len(params)):
                rhs.append(params[i] - stochgradparams[i])
            module.update_params(rhs)
        



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
    
    
    
class NearestUpsampling(Sequential):  
    def __init__(self, scale_factor, input_channels, output_channels, kernel_size, stride) -> None:
        super().__init__(NearestNeighbor(scale_factor), Conv2d(input_channels, output_channels, kernel_size, stride))
    
    def param(self):
        return self.args[1].param()
    
    def gradparam(self,idx):
        return self.args[1].gradparam(idx)
    
    def update_params(self, new_params):
        self.args[1].update_params(new_params)
        
        