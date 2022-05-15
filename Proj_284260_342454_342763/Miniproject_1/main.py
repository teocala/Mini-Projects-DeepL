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
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    # Using a GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using {} as device".format(device))


    # Load the data    
    print('Loading the data...')
    train_input, train_target = torch.load('train_data.pkl')
    test_input, test_target = torch.load('val_data.pkl')
    train_input, train_target = train_input.to(device), train_target.to(device)
    test_input, test_target = test_input.to(device), test_target.to(device)


    # Select a subset to speed up computations
    train_size = 50000
    train_input = train_input[:train_size]
    train_target = train_target[:train_size]
    test_input = test_input[:train_size]
    test_target = test_target[:train_size]

    # Data augmentation
    #trans = transforms.Compose([transforms.ColorJitter(brightness=.5,hue=.3),])
    flip_transform = transforms.RandomHorizontalFlip(p=1)
    #jitter = transforms.ColorJitter(brightness=.5,hue=.3)
    train_input = torch.cat([train_input, flip_transform(train_input)],0)
    train_target = torch.cat([train_target,flip_transform(train_target)],0)

    # Data normalization


    # Convert the data into float type
    train_input = train_input.float()
    train_target = train_target.float()
    test_input = test_input.float()
    test_target = test_target.float()

    
    print(f'Training data of size {train_input.shape}')

    # Defining and training the model
    model = Model()
    model = model.to(device)
    #model.load_pretrained_model()

    print('Training the model...')
    nb_epochs = 10
    model.train(train_input, train_target, num_epochs=nb_epochs)

    # Save the model
    path_to_model = "./best_model.pth"
    torch.save(model.state_dict(),path_to_model)


    # Visual comparison between original images and reconstructions
    # Just for us, probably we can't use matplotlib
    # Try to see if there are visualization tools only in Pytorch

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
        raise RuntimeError("Che cazzo hai messo coglione")


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


    # Delete training samples
    #del train_input,train_target
    #if device.type =="cuda":
        #torch.cuda.empty_cache()


    # Testing
    print('Using the trained model to denoise validation images...')
    with torch.no_grad():
        prediction = model.predict(test_input.to(device))

    # Evaluating error
    error = psnr(prediction, test_target)
    print(f'The PSNR on the validation set is {error} DB')

    stop = timeit.default_timer()
    print('Time: ', stop - start)


if __name__ == '__main__':
    main()