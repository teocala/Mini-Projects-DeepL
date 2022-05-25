import torch
from model import *
import pickle

import os
import sys

import matplotlib.pyplot as plt #TO REMOVE

os.chdir(sys.path[0])


def main():
    # Using a GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using {} as device".format(device))

    # Load the data    
    print('Loading the data...')
    train_input, train_target = torch.load('train_data.pkl')
    test_input, test_target = torch.load('val_data.pkl')


    # Select a subset to speed up computations
    train_size = 500
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

    # Save the model
    path_to_model = "./best_model.pth"
    outfile = open(path_to_model,'wb')
    pickle.dump(model, outfile)
    outfile.close()

    # Load the model
    model.load_pretrained_model()

    model.train(train_input, train_target)

    # Testing
    print('Using the trained model to denoise validation images...')
    with torch.no_grad():
        prediction = model.predict(test_input)




    # Visual comparison between original images and reconstructions
    # Just for us, probably we can't use matplotlib
    # Try to see if there are visualization tools only in Pytorch 

    # # Set "train" if you want to compare the training images, "test" for testing ones
    # images_to_visualize = "train"
    # if (images_to_visualize == "train"):
    #     with torch.no_grad():
    #         pred = model.predict(train_input).cpu() * 255
    #     train_input = train_input.cpu()
    #     images = train_input
    # elif (images_to_visualize == "test"):
    #     pred = prediction * 255
    #     test_input = test_input.cpu()
    #     images = test_input
    #     pred = pred.cpu()
    # else:
    #     raise RuntimeError("Che cazzo hai messo coglione")

    # n_comparisons = 5
    # for i in range(n_comparisons):
    #     f = plt.figure()
    #     f.add_subplot(1, 2, 1)
    #     fig1 = plt.imshow(images[i].int().permute(1, 2, 0))
    #     fig1.axes.get_xaxis().set_visible(False)
    #     fig1.axes.get_yaxis().set_visible(False)
    #     f.add_subplot(1, 2, 2)
    #     fig2 = plt.imshow(pred[i].int().permute(1, 2, 0))
    #     fig2.axes.get_xaxis().set_visible(False)
    #     fig2.axes.get_yaxis().set_visible(False)
    #     plt.show(block=True)



if __name__ == '__main__':
    main()