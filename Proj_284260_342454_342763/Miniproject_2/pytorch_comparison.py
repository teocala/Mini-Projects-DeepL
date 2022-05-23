# -*- coding: utf-8 -*-
import torch
from torch import nn

import os
import sys

import modules as mo

#os.chdir(sys.path[0])
from torch import set_grad_enabled
set_grad_enabled(True)





def train (model1, model2 , train_input , train_target ) -> None :
    batch_size = 100
    epochs = 10
    criterion1 = nn.MSELoss()
    criterion2 = mo.MSE()
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.0001)
    optimizer2 = mo.SGD(lr=0.0001)



    # to initialize same parameters for both the nets
    for i in range(len(model1._modules)):
        model2.args[i].update_params(model1._modules[str(i)]._parameters)
    


    for epoch in range(epochs):
        total_loss1 = 0
        total_loss2 = 0
        for input, targets in zip(train_input.split(batch_size), train_target.split(batch_size)):
            #input = convert_dim(input, [batch_size, 32, 16, 16])
            output1 = model1(input)
            output2 = model2.forward(input)
            #targets = convert_dim(targets, [batch_size, 3, 30, 30])
            
            loss1 = criterion1(output1, targets)
            loss2 = criterion2.forward(output2, targets)
            print(f'     Intermediate Losses {loss1} {loss2}')
            total_loss1 += loss1
            total_loss2 += loss2
            optimizer1.zero_grad()
            loss1.backward()
            gradx = criterion2.backward()
            gradx = model2.backward(gradx)
            optimizer1.step()
            optimizer2.step(model2)
        print(f'Epoch {epoch}/{epochs-1} Training Losses {total_loss1} {total_loss2}')




def convert_dim(targets, dims):
    tmp = torch.Tensor([dims])
    tmp = targets[:dims[0], :dims[1], :dims[2], :dims[3]]
    return tmp



def main():
    # Using a GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using {} as device".format(device))

    # Load the data    
    print('Loading the data...')
    train_input, train_target = torch.load('train_data.pkl')
    test_input, test_target = torch.load('val_data.pkl')


    # Select a subset to speed up computations
    train_size = 1000
    train_input = train_input[:train_size]
    train_target = train_target[:train_size]
    test_input = test_input[:train_size]
    test_target = test_target[:train_size]


    # Convert the data into float type
    train_input = train_input.float()
    train_target = train_target.float()
    test_input = test_input.float()
    test_target = test_target.float()


    
    
    class NearestUpsampling(nn.Module):
        def __init__(self, scale_factor, in_channels, out_channels, kernel_size, stride):
            super().__init__()
            self.UpsamplingNearest2d = nn.UpsamplingNearest2d(scale_factor=scale_factor)
            self.Conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
            self._parameters =  self.Conv2d._parameters
            
        def forward(self, input):
            x = self.UpsamplingNearest2d(input)
            return self.Conv2d(x)
      
            
    modelPT = nn.Sequential( # N, 3, 32, 32
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3),stride=2), # N, 32, 14, 14
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,1),stride=2), # N, 32, 6, 6
        nn.ReLU(),
        NearestUpsampling(scale_factor=2, in_channels = 32, out_channels = 32, kernel_size = (1,1), stride = 1),
        nn.ReLU(),
        NearestUpsampling(scale_factor=2, in_channels = 32, out_channels = 3, kernel_size = (1,1), stride = 1),
        nn.Sigmoid()
    )
    
    modelWE = mo.Sequential(# N, 3, 32, 32
        mo.Conv2d(input_channels = 3, output_channels = 32, kernel_size = (3,3), stride = 2), # N, 32, 14, 14
        mo.ReLU(),
        mo.Conv2d(input_channels = 32, output_channels = 32, kernel_size = (1,1), stride = 2), # N, 32, 8, 8
        mo.ReLU(),
        mo.NearestUpsampling(scale_factor = 2, input_channels = 32, output_channels = 32, kernel_size = (1,1), stride = 1), #  N, 32, 16, 16
        mo.ReLU(),
        mo.NearestUpsampling(scale_factor = 2, input_channels = 32, output_channels = 3, kernel_size = (1,1), stride = 1), # N, 3, 32, 32
        mo.Sigmoid()
    )
    

    print('Training the models...')
    train(modelPT, modelWE, train_input, train_target)



if __name__ == '__main__':
    main()
