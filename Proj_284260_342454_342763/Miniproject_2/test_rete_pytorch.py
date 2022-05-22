import torch
from torch import nn

import os
import sys

os.chdir(sys.path[0])

def train (model , train_input , train_target ) -> None :
    #: train˙input : tensor of size (N, C, H, W) containing a noisy version of the images.
    #: train˙target : tensor of size (N, C, H, W) containing another noisy version of the same images , which only differs from the input by their noise .
    batch_size = 100
    epochs = 10
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


    for epoch in range(epochs):
        total_loss = 0
        for input, targets in zip(train_input.split(batch_size), train_target.split(batch_size)):
            output = model(input)
            loss = criterion(output, targets)
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch}/{epochs-1} Training Loss {total_loss}')


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

    
    print(f'Training data of size {train_input.shape}')

    # Defining and training the model
    model = nn.Sequential( # N, 3, 32, 32
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3),stride=2), # N, 32, 14, 14
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,1),stride=2), # N, 32, 6, 6
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2), 
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (1,1), stride = 1),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2), 
            nn.Conv2d(in_channels = 32, out_channels = 3, kernel_size = (1,1), stride = 1),
            nn.Sigmoid()
    )

    # model = nn.Sequential( # N, 3, 32, 32
    #         nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3),stride=2),
    #         nn.Sigmoid(),
    # )
    # train_target_test = torch.randn((train_size,3,32,32)) # Just to work with the right dimension with one Conv2D

    print('Training the model...')
    train(model, train_input, train_target)

    # Testing
    print('Using the trained model to denoise validation images...')
    with torch.no_grad():
        prediction = model(test_input)



if __name__ == '__main__':
    main()