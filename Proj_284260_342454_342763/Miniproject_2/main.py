import torch
from model import *

import os
import sys


os.chdir(sys.path[0])

def compute_psnr(x, y, max_range=1.0):
        return 20 * torch.log10(torch.tensor(max_range)) - 10 * torch.log10(((x-y) ** 2).mean((1,2,3))).mean()

def main():
    # Using a GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using {} as device".format(device))

    # Load the data    
    print('Loading the data...')
    train_input, train_target = torch.load('train_data.pkl')
    test_input, test_target = torch.load('val_data.pkl')


    # Select a subset to speed up computations
    train_size = 50000
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
    model.train(train_input, train_target, 5)

    # # Save the model
    # model.save_pickle_state()

    # Load the model
    # model.load_pretrained_model()


    # model.train(train_input, train_target)

    # Testing
    print('Using the trained model to denoise validation images...')
    with torch.no_grad():
        prediction = model.predict(test_input) 


    print(f'psnr = {compute_psnr(prediction / 255.0, test_target / 255.0)}')
    print(f'reference psnr for input images = {compute_psnr(test_input / 255.0, test_target / 255.0)}')


if __name__ == '__main__':
    main()