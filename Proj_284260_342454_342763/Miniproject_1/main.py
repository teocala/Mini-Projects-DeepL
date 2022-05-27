import torch
from model import *
from others.utilities import *
from torchvision import transforms
import os
import sys
import timeit
import matplotlib.pyplot as plt
import numpy as np
import random

os.chdir(sys.path[0])


def main():
    start = timeit.default_timer()

    '''
    # To reproduce our results
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    '''

    # Load the data    
    print('Loading the data...')
    train_input, train_target = torch.load('train_data.pkl')
    test_input, test_target = torch.load('val_data.pkl')

    print(f'Training data of size {train_input.shape}')

    # Defining and training the model
    model = Model()

    nb_epochs = 30
    model.train(train_input, train_target, num_epochs=nb_epochs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_input = test_input.float()
    test_target = test_target.float()
    test_input = test_input.to(device)
    test_target = test_target.to(device)


    model.load_pretrained_model()
    model.to(device)
    model.unet.eval()
    # Visual comparison between original images and reconstructions
    # Set "train" if you want to compare the training images, "test" for testing ones
    images_to_visualize = "test"
    #Set the number of images you want to see
    n_comparisons = 30
    

    if (images_to_visualize == "train"):
        input = train_input
        test = train_target
    elif (images_to_visualize == "test"):
        input = test_input
        test = test_target
    else:
        raise RuntimeError("Invalid split")


    for i in range(n_comparisons):
        pred = model.predict(input[i].unsqueeze(0)).cpu()
        f = plt.figure()
        f.add_subplot(1, 3, 1)
        fig1 = plt.imshow(input[i].cpu().int().permute(1, 2, 0))
        fig1.axes.get_xaxis().set_visible(False)
        fig1.axes.get_yaxis().set_visible(False)
        f.add_subplot(1, 3, 2)
        fig2 = plt.imshow(pred[0].int().permute(1, 2, 0))
        fig2.axes.get_xaxis().set_visible(False)
        fig2.axes.get_yaxis().set_visible(False)
        f.add_subplot(1,3,3)
        fig3 = plt.imshow(test[i].cpu().int().permute(1,2,0))
        fig3.axes.get_xaxis().set_visible(False)
        fig3.axes.get_yaxis().set_visible(False)
        plt.show(block=True)


    # Testing
    print('Using the trained model to denoise validation images...')
    with torch.no_grad():
        prediction = model.predict(test_input.to(device))

    # Evaluating error
    error = compute_psnr(prediction, test_target)
    print(f'The PSNR on the validation set is {error} DB')

    stop = timeit.default_timer()
    print('Time: ', stop - start)


if __name__ == '__main__':
    main()