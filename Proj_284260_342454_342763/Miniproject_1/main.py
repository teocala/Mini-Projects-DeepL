import torch
from model import *
from others.utilities import *

import os
import sys

os.chdir(sys.path[0])


def main():
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
    model = Model()
    print('Training the model...')
    model.train(train_input, train_target)

    # Testing
    print('Using the trained model to denoise validation images...')
    prediction = model.predict(test_input)

    # Evaluating error
    error = psnr(prediction, test_target)
    print(f'The PSNR on the validation set is {error} DB')

if __name__ == '__main__':
    main()