import torch 
from torch import nn
from others.utilities import *

### For mini-project 1
class Model(nn.Module):
    def __init__(self) -> None:
        ## instantiate model + optimizer + loss function + any other stuff you need
        super().__init__()

        '''
        # Network with larger kernel size 
        self.encoder = nn.Sequential( # initial: N, 3, 32, 32
            nn.Conv2d(3, 32, kernel_size = 5, stride = 1), # N, 32, 28, 28
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 5, stride = 1, padding = 2), # N, 64, 28, 28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0), # N, 64, 14, 14
            nn.Conv2d(64, 64, kernel_size = 5, stride = 1, padding = 2), # N, 64, 14, 14
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size = 5, stride = 1), # N, 128, 10, 10
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0), # N, 128, 5, 5
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size = 5, stride = 1, padding = 2), # N, 64, 5, 5
            nn.ReLU(),
            nn.Upsample(scale_factor=2,mode = 'bilinear', align_corners=True), # N, 64, 10, 10
            nn.ConvTranspose2d(64, 64, kernel_size = 5, stride = 1, padding = 2), # N, 64, 10, 10
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size = 5, stride = 1), # N, 64, 14, 14
            nn.ReLU(),
            nn.Upsample(scale_factor=2,mode = 'bilinear', align_corners=True), # N, 3, 28, 28
            nn.ConvTranspose2d(3, 3, kernel_size = 5, stride = 1), # N, 3, 32, 32
            nn.ReLU()
        )
        '''

        '''
        # Network with smaller kernel size
        self.encoder = nn.Sequential(  # initial: N, 3, 32, 32
            nn.Conv2d(3, 32, kernel_size=5, stride=1),  # N, 32, 28, 28
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # N, 64, 26, 26
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # N, 64, 24, 24
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # N, 128, 22, 22
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),  # N, 128, 20, 20
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # N, 128, 10, 10
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1),  # N, 64, 12, 12
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # N, 64, 24, 24
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1),  # N, 64, 26, 26
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1),  # N, 64, 28, 28
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1),  # N, 64, 30, 30
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1),  # N, 3, 32, 32
            nn.ReLU()

        )
        '''

        '''
        # Network with max-pool and up-sample in the middle
        # psnr 20.10

        self.encoder = nn.Sequential(  # initial: N, 3, 32, 32
            nn.Conv2d(3, 32, kernel_size=5, stride=1),  # N, 32, 28, 28
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=1),  # N, 64, 24, 24
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # N, 64, 22, 22
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),  # N, 128, 20, 20
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # N, 128, 10, 10
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1),  # N, 64, 12, 12
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1),  # N, 64, 14, 14
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # N, 64, 28, 28
            nn.ConvTranspose2d(32, 3, kernel_size=5, stride=1),  # N, 3, 32, 32
            nn.ReLU()

        )
        '''

        '''
        # Network with max-pool and up-sample at the end and beginning
        # Network with same structure for encoder and decoder
        # psnr 21.09 if 40 epochs

        self.encoder = nn.Sequential(  # initial: N, 3, 32, 32
            nn.Conv2d(3, 32, kernel_size=5, stride=1),  # N, 32, 28, 28
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=1),  # N, 64, 24, 24
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # N, 64, 22, 22
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),  # N, 128, 20, 20
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # N, 128, 10, 10
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # N, 64, 20, 20
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1),  # N, 64, 22, 22
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1),  # N, 64, 24, 24
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=1),  # N, 64, 28, 28
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=5, stride=1),  # N, 3, 32, 32
            nn.ReLU()

        )
        '''

        '''
        # Network with max-pool and up-sample at the end and beginning
        # Network with same structure of decoder and encoder
        # No activation function at the end (inspiration from the slides)
        # psnr 19.01 with 40 epochs
        # I have then repeated adding a final activation function both on encoder and decoder 
        # The new result is psnr 18.01 with 20 epochs

        self.encoder = nn.Sequential(  # initial: N, 3, 32, 32
            nn.Conv2d(3, 32, kernel_size=5, stride=1),  # N, 32, 28, 28
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=1),  # N, 64, 24, 24
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # N, 64, 22, 22
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),  # N, 128, 20, 20
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # N, 128, 10, 10
            nn.Conv2d(128,128,kernel_size=3, stride=1),   #N, 128, 8, 8
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1),  # N, 64, 10, 10
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # N, 64, 20, 20
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1),  # N, 64, 22, 22
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1),  # N, 64, 22, 22
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=1),  # N, 64, 28, 28
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=5, stride=1),  # N, 3, 32, 32

        )
        '''

        '''
        # Thomas' best model
        # psnr 20.95
        # Without final activation function psnr 21.27

        self.encoder = nn.Sequential(  # initial: N, 3, 32, 32
            nn.Conv2d(3, 32, kernel_size=3, stride=1),  # N, 32, 30, 30
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # N, 64, 28, 28
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1),  # N, 64, 24, 24
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=1),  # N, 128, 20, 20
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # N, 128, 10, 10
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=1),  # N, 64, 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=1, padding=2),  # N, 64, 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=1),  # N, 32, 15, 15
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # N, 64, 30, 30
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1),  # N, 3, 32, 32
        )
        '''

        '''
        # Matteo's best model
        # With activation function psnr 22.48
        # Without activation function psnr 22.68

        self.encoder = nn.Sequential(  # initial: N, 3, 32, 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # N, 32, 32, 32
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),  # N, 32, 16, 16
            nn.Conv2d(32, 64, kernel_size=2),  # N, 64, 15, 15
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),  # N, 64, 13, 13
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, padding=2),  # N, 64, 13, 13
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3),  # N, 32, 15, 15
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # N, 32, 30, 30
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3),  # N, 3, 32, 32
        )
        '''


        '''
        # Use of a Median Pooling, batch size 1000 and without final activation function, psnr 22.66
        # Final Activation function in the decoder and batch size 100 , psnr 22.85
        # Clamping at the end of the predict, psnr 23.31
        # Using SmoothL1Loss , psnr 23.11
        # Get back to average pooling and normal loss (with clamping), psnr 23.60

        self.encoder = nn.Sequential(  # initial: N, 3, 32, 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # N, 32, 32, 32
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),  # N, 32, 16, 16
            nn.Conv2d(32, 64, kernel_size=2),  # N, 64, 15, 15
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),  # N, 64, 13, 13
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, padding=2),  # N, 64, 13, 13
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3),  # N, 32, 15, 15
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # N, 32, 30, 30
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3),  # N, 3, 32, 32
            nn.ReLU()
        )
        '''

        '''
        # Median pooling at the beginning and modification to adapt , psnr 21.44
        self.encoder = nn.Sequential(  # initial: N, 3, 32, 32
            MedianPool2d(kernel_size=2, stride=2, padding=0),  # N, 32, 16, 16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # N, 32, 16, 16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),  # N, 64, 14, 14
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),  # N, 64, 12, 12
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, padding=2),  # N, 64, 12, 12
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3),  # N, 32, 14, 14
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # N, 32, 28, 28
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=5),  # N, 3, 32, 32
            nn.ReLU()
        )
        '''

        '''
        # Median pooling at the beginning with padding, to eliminate noise at the beginning, psnr 22.79
        # Added a clamp operation at the end of the predict function
        # Without final activation function in the decoder , psnr 22.77

        self.encoder = nn.Sequential(  # initial: N, 3, 32, 32
            MedianPool2d(kernel_size=2, stride=1, padding=1),  # N, 3, 32, 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # N, 32, 32, 32
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),  # N, 32, 16, 16
            nn.Conv2d(32, 64, kernel_size=3),  # N, 64, 14, 14
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),  # N, 64, 12, 12
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, padding=2),  # N, 64, 12, 12
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3),  # N, 32, 14, 14
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # N, 32, 28, 28
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=5),  # N, 3, 32, 32
        )
        '''


        '''
        # Median pooling after the first convolution, psnr

        self.encoder = nn.Sequential(  # initial: N, 3, 32, 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # N, 32, 32, 32
            nn.ReLU(),
            MedianPool2d(kernel_size=2, stride=1, padding=1),  # N, 3, 32, 32
            nn.Conv2d(32, 64, kernel_size=3),  # N, 64, 30, 30
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),  # N, 32, 15, 15
            nn.Conv2d(64, 128, kernel_size=3, padding=0),  # N, 64, 13, 13
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, padding=2),  # N, 64, 13, 13
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3),  # N, 32, 15, 15
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # N, 32, 30, 30
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3),  # N, 3, 32, 32
        )
        '''

        '''
        # First U-Net tried, but there is something to improve
        self.enc_chs = (3,64,128)  # N , 128, 5 , 5
        self.dec_chs = (128,64)
        self.encoder = Encoder(self.enc_chs)
        self.decoder = Decoder(self.dec_chs)
        self.head = nn.Conv2d(self.dec_chs[-1],3,3,1,1)
        self.retain_dim = True
        '''

        '''
        # New implementation of the U-Net 
        # Obtained psnr of 24.4 
        # Try with fewer epochs, after 10 epochs the loss stay almost the same 
        #self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.down_conv_1 = double_conv(3,64)
        self.down_sample_1 = nn.MaxPool2d(kernel_size=2 ,stride=2)
        self.down_conv_2 = double_conv(64, 128)
        self.down_sample_2 = nn.MaxPool2d(kernel_size=2,stride = 2)
        self.down_conv_3 = double_conv(128,256)
        self.down_sample_3 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.down_conv_4 = double_conv(256,512)
        self.down_sample_4 = nn.MaxPool2d(kernel_size=2, stride=2)



        self.up_sample_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=3, padding = 1, stride=1)
        self.up_sample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv_2 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.up_sample_3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv_3 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.up_sample_4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1)

        self.out = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)
        '''

        '''
        # New trial of the U-Net
        # No use of the maxpooling and upsample 
        # psnr of 23.19, 24.22 with double channels 
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1)  #N , 32, 28,28   --->
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=32, kernel_size=5, stride=1)  # N , 32, 24 , 24
        self.conv3= nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1) # N, 64, 19, 19  --->
        self.conv4 = nn.Conv2d(in_channels=64,out_channels=128, kernel_size=5, stride=1)  # N, 128, 14, 14
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1) # N, 128, 11, 11 --->
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1) # N , 128, 8, 8
        self.relu = nn.ReLU(inplace=True)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=1) # N, 256, 11, 11
        # Concatenate along channels
        self.upconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1) # N , 128, 14, 14
        self.upconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=1) # N, 64, 19, 19
        #Concatenate
        self.upconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=1) # N, 64, 24, 24
        self.upconv5 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=1) # N , 32, 29, 29
        #Concatenate
        self.upconv6 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=1) #N , 64, 32, 32

        self.out = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, stride=1) # N , 3 , 32, 32
        '''

        '''
        # Third trial, based on the best network without U-net
        # psnr of 24.4

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3,32,kernel_size=3,padding=1) # N, 32, 32, 32
        self.avgpool = nn.AvgPool2d(kernel_size=2,stride=2,padding=0) # N, 32, 16, 16
        self.conv2 = nn.Conv2d(32,64, kernel_size=3) # N, 64, 14 ,14
        self.conv3 = nn.Conv2d(64,128, kernel_size=3, padding=0) # N, 128, 12, 12
        self.conv4 = nn.Conv2d(128,256, kernel_size=3, padding=0) # N , 256, 10 , 10

        self.upconv4 = nn.ConvTranspose2d(256,128, kernel_size=3,padding=0) # N , 128, 12, 12
        self.upconv3 = nn.ConvTranspose2d(256,64, kernel_size=3, padding=0) # N , 64, 14, 14
        self.upconv2 = nn.ConvTranspose2d(128,32,kernel_size=3, padding=0) # N , 32, 16, 16
        self.upsample = nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=True) # N , 32, 32, 32

        self.conv_final = nn.Conv2d(64,64,kernel_size=3, padding=1)
        self.out = nn.Conv2d(64,3,kernel_size=1)
        '''

        '''
        # Fourth trial based on the previous one
        # Strided convolution ad transpose convolution instead of avg pooling and upsample
        # Little improvement, 24.48

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # N, 32, 32, 32
        self.down = nn.Conv2d(32,32,kernel_size=3, stride=2, padding=1)  # N, 32, 16, 16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # N, 64, 14 ,14
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=0)  # N, 128, 12, 12
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # N , 256, 10 , 10

        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=0)  # N , 128, 12, 12
        self.upconv3 = nn.ConvTranspose2d(256, 64, kernel_size=3, padding=0)  # N , 64, 14, 14
        self.upconv2 = nn.ConvTranspose2d(128, 32, kernel_size=3, padding=0)  # N , 32, 16, 16
        self.up = nn.ConvTranspose2d(32,32,kernel_size=4, stride=2, padding=1)  # N , 32, 32, 32

        self.conv_final = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.out = nn.Conv2d(64, 3, kernel_size=1)
        '''

        '''
        # Fifth trial
        # Using also ResNet 
        # psnr of 24.53
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # N, 32, 32,
        self.resnet1 = ResNetBlock(nb_channels=32, kernel_size=3, batch_normalization=True, skip_connections=True)
        self.down = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)  # N, 32, 16, 16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # N, 64, 14 ,14
        self.resnet2 = ResNetBlock(nb_channels=64, kernel_size=3, batch_normalization=True, skip_connections=True)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=0)  # N, 128, 12, 12
        self.resnet3 = ResNetBlock(nb_channels=128, kernel_size=3, batch_normalization=True, skip_connections=True)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # N , 256, 10 , 10

        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=0)  # N , 128, 12, 12
        self.resnet4 = ResNetBlock(nb_channels=128,kernel_size=3,batch_normalization=True,skip_connections=True)
        self.upconv3 = nn.ConvTranspose2d(256, 64, kernel_size=3, padding=0)  # N , 64, 14, 14
        self.resnet5 = ResNetBlock(nb_channels=64,kernel_size=3,batch_normalization=True,skip_connections=True)
        self.upconv2 = nn.ConvTranspose2d(128, 32, kernel_size=3, padding=0)  # N , 32, 16, 16
        self.resnet6 = ResNetBlock(nb_channels=32,kernel_size=3,batch_normalization=True,skip_connections=True)
        self.up = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)  # N , 32, 32, 32

        self.conv_final = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.out = nn.Conv2d(64, 3, kernel_size=1)
        '''

        '''
        # Sixth trial
        # psnr 24.58 
        # Final ResNet without batch_normalization

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # N, 32, 32,
        self.resnet1 = ResNetBlock(nb_channels=32, kernel_size=3, batch_normalization=True, skip_connections=True)
        self.down = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)  # N, 32, 16, 16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # N, 64, 14 ,14
        self.resnet2 = ResNetBlock(nb_channels=64, kernel_size=3, batch_normalization=True, skip_connections=True)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=0)  # N, 128, 12, 12
        self.resnet3 = ResNetBlock(nb_channels=128, kernel_size=3, batch_normalization=True, skip_connections=True)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # N , 256, 10 , 10

        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=0)  # N , 128, 12, 12
        self.resnet4 = ResNetBlock(nb_channels=128, kernel_size=3, batch_normalization=True, skip_connections=True)
        self.upconv3 = nn.ConvTranspose2d(256, 64, kernel_size=3, padding=0)  # N , 64, 14, 14
        self.resnet5 = ResNetBlock(nb_channels=64, kernel_size=3, batch_normalization=True, skip_connections=True)
        self.upconv2 = nn.ConvTranspose2d(128, 32, kernel_size=3, padding=0)  # N , 32, 16, 16
        self.resnet6 = ResNetBlock(nb_channels=32, kernel_size=3, batch_normalization=True, skip_connections=True)
        self.up = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)  # N , 32, 32, 32

        self.conv_final = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.resnet7 = ResNetBlock(nb_channels=64,kernel_size=3,batch_normalization=False, skip_connections=True)

        self.out= nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.resnet8 = ResNetBlock(nb_channels=3,kernel_size=3,batch_normalization=False,skip_connections=True)
        '''

        '''
        # Seventh trial 
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # N, 32, 32, 32
        self.down = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)  # N, 32, 16, 16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # N, 64, 14 ,14
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=0)  # N, 128, 12, 12
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # N , 256, 10 , 10
        self.conv5 = nn.Conv2d(256,256,kernel_size=3, padding=0) # N, 256, 8, 8
        self.conv6 = nn.Conv2d(256,256, kernel_size=3, padding=0)  # N , 256, 6, 6
        self.conv7 = nn.Conv2d(256,256, kernel_size=3, padding=0) # N , 256 , 4 ,4

        self.upconv7 = nn.ConvTranspose2d(256, 256, kernel_size=3, padding=0)  # N , 256, 6, 6
        self.upconv6 = nn.ConvTranspose2d(512, 256, kernel_size=3, padding=0)  # N , 256, 8, 8
        self.upconv5 = nn.ConvTranspose2d(512,256, kernel_size=3, padding=0)  # N , 256, 10, 10
        self.upconv4 = nn.ConvTranspose2d(512,128,kernel_size=3,padding=0) # N, 128 , 12, 12
        self.upconv3 = nn.ConvTranspose2d(256,64, kernel_size=3, padding=0) # N , 64, 14, 14
        self.upconv2 = nn.ConvTranspose2d(128,32,kernel_size=3,padding=0) # N , 32, 16, 16
        self.up = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)  # N , 32, 32, 32

        self.conv_final = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.out = nn.Conv2d(64, 3, kernel_size=1)
        self.resnet1 = ResNetBlock(nb_channels=3, kernel_size=3, batch_normalization=False, skip_connections=True)
        self.resnet2 = ResNetBlock(nb_channels=3, kernel_size=3, batch_normalization=False, skip_connections=True)
        '''

        '''
        # Try to get the real U-Net with sampling modified 
        # psnr of 24.45
        self.down_conv_1 = double_conv(3, 32)  # N , 32, 32, 32
        self.down_sample_1 = nn.Conv2d(32,32,kernel_size=3,padding=1,stride=2) # N , 32, 16 , 16
        self.down_conv_2 = double_conv(32, 64)    # N, 64,16,16
        self.down_sample_2 = nn.Conv2d(64,64,kernel_size=3,padding=1,stride=2)   # N , 64, 8, 8
        self.down_conv_3 = double_conv(64, 128)                                  # N, 128, 8,8
        self.down_sample_3 = nn.Conv2d(128,128,kernel_size=3,padding=1,stride=2)   # N, 128, 4 ,4
        self.down_conv_4 = double_conv(128, 256)     # N , 256, 4 , 4
        self.down_sample_4 = nn.Conv2d(256,256,kernel_size=3, padding=1, stride=2)    # N , 256, 2, 2
        self.down_conv_5 = double_conv(256,512)   # N , 512, 1, 1

        self.up_sample_0 = nn.ConvTranspose2d(512,256,kernel_size=4, padding=1, stride=2)  # N , 256, 4, 4
        self.up_conv_1 = double_conv(512,256)    # N , 256, 4, 4
        self.up_sample_1 = nn.ConvTranspose2d(256,128,kernel_size=4,padding=1,stride=2) # N , 128, 4, 4
        self.up_conv_2 = double_conv(256,128)   # N, 128, 4 , 4
        self.up_sample_2 = nn.ConvTranspose2d(128,64, kernel_size=4, padding=1, stride=2) # N , 64, 8, 8
        self.up_conv_3 = double_conv(128,64)  # N , 64, 8, 8
        self.up_sample_3 = nn.ConvTranspose2d(64,32,kernel_size=4,padding=1,stride=2) # N , 32, 16, 16
        self.up_conv_4 = double_conv(64,32)   # N , 32, 16, 16

        self.resnet1 = ResNetBlock(nb_channels=32, kernel_size=3, batch_normalization=False, skip_connections=True)
        self.resnet2 = ResNetBlock(nb_channels=32, kernel_size=3, batch_normalization=False, skip_connections=True)

        self.out = nn.Conv2d(32,3,kernel_size=1)
        '''

        '''
        # U-Net with addition and not concatenation
        # Psnr of 24.45 
        self.down_conv_1 = double_conv(3, 32)  # N , 32, 32, 32
        self.down_sample_1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2)  # N , 32, 16 , 16
        self.down_conv_2 = double_conv(32, 64)  # N, 64,16,16
        self.down_sample_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2)  # N , 64, 8, 8
        self.down_conv_3 = double_conv(64, 128)  # N, 128, 8,8
        self.down_sample_3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2)  # N, 128, 4 ,4
        self.down_conv_4 = double_conv(128, 256)  # N , 256, 4 , 4
        self.down_sample_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)  # N , 256, 2, 2
        self.down_conv_5 = double_conv(256, 512)  # N , 512, 1, 1

        self.up_sample_0 = nn.ConvTranspose2d(512, 256, kernel_size=4, padding=1, stride=2)  # N , 256, 2, 2
        self.up_conv_1 = double_conv(256, 256)  # N , 256, 2, 2
        self.up_sample_1 = nn.ConvTranspose2d(256, 128, kernel_size=4, padding=1, stride=2)  # N , 128, 4, 4
        self.up_conv_2 = double_conv(128, 128)  # N, 128, 4 , 4
        self.up_sample_2 = nn.ConvTranspose2d(128, 64, kernel_size=4, padding=1, stride=2)  # N , 64, 8, 8
        self.up_conv_3 = double_conv(64, 64)  # N , 64, 8, 8
        self.up_sample_3 = nn.ConvTranspose2d(64, 32, kernel_size=4, padding=1, stride=2)  # N , 32, 16, 16
        self.up_conv_4 = double_conv(32, 32)  # N , 32, 16, 16

        self.out = nn.Conv2d(32, 3, kernel_size=1)
        '''

        '''
        # Modifications on the previous one
        self.down_conv_1 = double_conv(3, 32)  # N , 32, 32, 32
        #self.resnet1 = ResNetBlock(nb_channels=32,kernel_size=3,batch_normalization=False,skip_connections=True)
        self.down_sample_1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2)  # N , 32, 16 , 16
        self.down_conv_2 = double_conv(32, 64)  # N, 64,16,16
        #self.resnet2 = ResNetBlock(nb_channels=64,kernel_size=3,batch_normalization=False,skip_connections=True)
        self.down_sample_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2)  # N , 64, 8, 8
        self.down_conv_3 = double_conv(64, 128)  # N, 128, 8,8
        #self.resnet3 = ResNetBlock(nb_channels=128, kernel_size=3, batch_normalization=False,skip_connections=True)
        self.down_sample_3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2)  # N, 128, 4 ,4
        self.down_conv_4 = double_conv(128, 256)  # N , 256, 4 , 4
        #self.resnet4 = ResNetBlock(nb_channels=256,kernel_size=3,batch_normalization=False,skip_connections=True)
        self.down_sample_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)  # N , 256, 2, 2
        self.down_conv_5 = double_conv(256, 512)  # N , 512, 1, 1

        self.up_sample_0 = nn.ConvTranspose2d(512, 256, kernel_size=4, padding=1, stride=2)  # N , 256, 4, 4
        self.up_conv_1 = double_conv(256, 256)  # N , 256, 4, 4
        #self.resnet5 = ResNetBlock(nb_channels=256,kernel_size=3, batch_normalization=False,skip_connections=True)
        self.up_sample_1 = nn.ConvTranspose2d(256, 128, kernel_size=4, padding=1, stride=2)  # N , 128, 8, 8
        self.up_conv_2 = double_conv(128, 128)  # N, 128, 8 , 8
        #self.resnet6 = ResNetBlock(nb_channels=128, kernel_size=3, batch_normalization=False, skip_connections=True)
        self.up_sample_2 = nn.ConvTranspose2d(128, 64, kernel_size=4, padding=1, stride=2)  # N , 64, 16, 16
        self.up_conv_3 = double_conv(64, 64)  # N , 64, 32, 32
        #self.resnet7 = ResNetBlock(nb_channels=64,kernel_size=3,batch_normalization=False, skip_connections=True)
        self.up_sample_3 = nn.ConvTranspose2d(64, 32, kernel_size=4, padding=1, stride=2)  # N , 32, 32, 32
        self.up_conv_4 = double_conv(32, 32)  # N , 32, 32, 32

        #self.resnet8 = ResNetBlock(32,3,False,True)
        self.out = nn.Conv2d(32, 3, kernel_size=1)
        '''

        # U - net with more latent features
        # Faster, can use more epochs
        # Using 20 epochs, Data augmentation and batch size 20 ---> psnr 24.65
        self.down_conv_1 = double_conv(3, 32)  # N , 32, 32, 32
        self.down_sample_1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2)  # N , 32, 16 , 16
        self.down_conv_2 = double_conv(32, 64)  # N, 64,16,16



        #self.down_sample_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2)  # N , 64, 8, 8
        #self.down_conv_3 = double_conv(64, 128)  # N, 128, 8,8

        #self.down_sample_3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2)  # N, 128, 4 ,4
        #self.down_conv_4 = double_conv(128, 256)  # N , 256, 4 , 4
        #self.down_sample_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)  # N , 256, 2, 2
        #self.down_conv_5 = double_conv(256, 512)  # N , 512, 2, 2
        #self.down_sample_5 = nn.Conv2d(512,512,kernel_size=3, padding=1, stride=2)  # N , 512, 1, 1
        #self.down_conv_6 = double_conv(512,1024) # N , 1024, 1 , 1

        #self.up_conv_0 = double_conv(1024,1024)  # N , 1024, 1 , 1
        #self.up_sample_0 = nn.ConvTranspose2d(1024, 512, kernel_size=4, padding=1, stride=2)  # N , 512, 2, 2
        #self.up_conv_1 = double_conv(1024, 512)  # N , 512, 2, 2
        #self.up_sample_1 = nn.ConvTranspose2d(512, 256, kernel_size=4, padding=1, stride=2)  # N , 256, 4, 4
        #self.up_conv_2 = double_conv(256, 256)  # N, 256, 4 , 4
        #self.up_sample_2 = nn.ConvTranspose2d(256, 128, kernel_size=4, padding=1, stride=2)  # N , 128, 8, 8

        #self.up_conv_3 = double_conv(128, 128)  # N , 128, 8, 8
        #self.up_sample_3 = nn.ConvTranspose2d(128, 64, kernel_size=4, padding=1, stride=2)  # N , 64, 16, 16


        self.up_conv_4 = double_conv(64, 64)  # N , 64, 16, 16
        self.up_sample_4 = nn.ConvTranspose2d(64,32,kernel_size=4,padding=1,stride=2)  #N, 32, 32, 32
        self.up_conv_5 = double_conv(64,32)  # N , 32, 32, 32


        self.out = nn.Conv2d(32, 3, kernel_size=1)

        #self.resnet1 = ResNetBlock(nb_channels=64,kernel_size=3,batch_normalization=False, skip_connections=True)
        #self.resnet2 = ResNetBlock(nb_channels=32,kernel_size=3,batch_normalization=False,skip_connections=True)
        self.sigmoid = nn.Sigmoid()


        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.00085)
        

    def load_pretrained_model(self ) -> None:
        ## This loads the parameters saved in bestmodel . pth into the model
        checkpoint = torch.load("./best_model.pth")
        self.load_state_dict(checkpoint)



    def train(self, train_input, train_target, num_epochs) -> None:
        # : train˙input : tensor of size (N , C , H , W ) containing a noisy version of the images.
        # : train˙target : tensor of size (N , C , H , W ) containing another noisy version of the
        # same images , which only differs from the input by their noise.

        batch_size = 20

        for epoch in range(num_epochs):
            total_loss = 0
            for batch_input, batch_target in zip(train_input.split(batch_size), train_target.split(batch_size)):
                output = self.predict(batch_input)
                loss = self.criterion(output, batch_target)
                total_loss += loss*batch_size
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            total_loss /= train_input.size(0)
            print(f'Epoch {epoch}/{num_epochs - 1} Training Loss {total_loss}')



    def predict(self, test_input) -> torch.Tensor:
        # : test˙input : tensor of size ( N1 , C , H , W ) that has to be denoised by the trained
        # or the loaded network .
        # : returns a tensor of the size ( N1 , C , H , W )
        
        # like the forward method

        #y = self.encoder(test_input)
        #y = self.decoder(y)
        #y = torch.clamp(y,0,255)

        '''
        enc_ftrs = self.encoder(test_input)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)

        if self.retain_dim:
            out = F.interpolate(out, 32)

        out = torch.clamp(out,0,255)
        '''

        ''' 
        #Encoder
        x1 = self.down_conv_1(test_input)  # N , 64, 32, 32    --->
        x2 = self.down_sample_1(x1)  # N , 64, 16, 16
        x3 = self.down_conv_2(x2)  # N , 128, 16, 16  --->
        x4 = self.down_sample_2(x3)  #N, 128, 8, 8
        x5 = self.down_conv_3(x4)  #N , 256, 8, 8  --->
        x6 = self.down_sample_3(x5) #N, 256, 4, 4
        x7 = self.down_conv_4(x6) # N , 512, 2, 2 --->
        x8 = self.down_sample_4(x7) # N, 512, 1, 1

        #Decoder
        x = self.up_sample_1(x8) # N , 512, 2, 2
        x = self.up_conv_1(torch.cat([x,x7],1)) #N, 256, 4, 4
        x = self.up_sample_2(x)  # N , 256, 8, 8
        x = self.up_conv_2(torch.cat([x,x5],1)) # N, 128, 16, 16
        x = self.up_sample_3(x) # N , 128, 16, 16
        x = self.up_conv_3(torch.cat([x,x3],1)) # N , 64, 16, 16
        x = self.up_sample_3(x)  # N , 64, 32, 32
        x = self.up_conv_4(torch.cat([x,x1],1)) # N, 64, 32,32

        x = self.out(x)

        x = torch.clamp(x,0,255)
        '''

        '''
        # Second trial of the U-net
        # Encoder
        x1 = self.relu(self.conv1(test_input))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(x4))
        x6 = self.conv6(x5)

        x = self.relu(self.upconv1(x6))
        x = self.relu(self.upconv2(torch.cat((x,x5),1)))
        x = self.relu(self.upconv3(x))
        x = self.relu(self.upconv4( torch.cat((x,x3),1) ) )
        x = self.relu(self.upconv5(x))
        x = self.upconv6( torch.cat((x,x1),1) )

        x = self.out(x)
        x = self.relu(x)
        x = torch.clamp(x,0,255)
        '''

        '''
        # third trial of U-net 
        x1 = self.relu(self.conv1(test_input))
        x1_down = self.avgpool(x1)

        x2 = self.relu(self.conv2(x1_down))
        x3 = self.relu(self.conv3(x2))
        x4 = self.conv4(x3)

        x = self.relu(self.upconv4(x4))
        x = self.relu( self.upconv3(torch.cat((x,x3),1)))
        x = self.relu( self.upconv2( torch.cat((x,x2),1)))
        x = self.upsample(x)

        x = self.relu( self.conv_final(torch.cat((x,x1),1)))
        x = self.out(x)
        x = self.relu(x)

        x = torch.clamp(x,0,255)
        '''

        '''
        # Fourth trial
        x1 = self.relu(self.conv1(test_input))
        x1_down = self.down(x1)

        x2 = self.relu(self.conv2(x1_down))
        x3 = self.relu(self.conv3(x2))
        x4 = self.conv4(x3)

        x = self.relu(self.upconv4(x4))
        x = self.relu(self.upconv3(torch.cat((x, x3), 1)))
        x = self.relu(self.upconv2(torch.cat((x, x2), 1)))
        x = self.up(x)

        x = self.relu(self.conv_final(torch.cat((x, x1), 1)))
        x = self.out(x)
        x = self.relu(x)

        x = torch.clamp(x, 0, 255)
        '''

        '''
        # Fifth trial
        x1 = self.relu(self.conv1(test_input))
        x1_r = self.resnet1(x1)
        x1_down = self.down(x1_r)

        x2 = self.relu(self.conv2(x1_down))
        x2_r = self.resnet2(x2)
        x3 = self.relu(self.conv3(x2_r))
        x3_r = self.resnet3(x3)
        x4 = self.conv4(x3_r)

        x = self.relu(self.upconv4(x4))
        x = self.resnet4(x)
        x = self.relu(self.upconv3(torch.cat((x, x3), 1)))
        x = self.resnet5(x)
        x = self.relu(self.upconv2(torch.cat((x, x2), 1)))
        x = self.resnet6(x)
        x = self.up(x)

        x = self.relu(self.conv_final(torch.cat((x, x1), 1)))
        x = self.out(x)
        x = self.relu(x)

        x = torch.clamp(x, 0, 255)
        '''

        '''
        # Sixth trial
        x1 = self.relu(self.conv1(test_input))
        x1_r = self.resnet1(x1)
        x1_down = self.down(x1_r)

        x2 = self.relu(self.conv2(x1_down))
        x2_r = self.resnet2(x2)
        x3 = self.relu(self.conv3(x2_r))
        x3_r = self.resnet3(x3)
        x4 = self.conv4(x3_r)

        x = self.relu(self.upconv4(x4))
        x = self.resnet4(x)
        x = self.relu(self.upconv3(torch.cat((x, x3), 1)))
        x = self.resnet5(x)
        x = self.relu(self.upconv2(torch.cat((x, x2), 1)))
        x = self.resnet6(x)
        x = self.up(x)

        x = self.relu(self.conv_final(torch.cat((x, x1), 1)))
        x = self.resnet7(x)
        x = self.relu(self.out(x))
        x = self.resnet8(x)


        x = torch.clamp(x, 0, 255)
        '''

        '''
        # Seventh trial
        x1 = self.relu(self.conv1(test_input))
        x1_down = self.down(x1)

        x2 = self.relu(self.conv2(x1_down))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(x4))
        x6 = self.relu(self.conv6(x5))
        x7 = self.conv7(x6)

        x = self.relu(self.upconv7(x7))
        x = self.relu(self.upconv6(torch.cat((x, x6), 1)))
        x = self.relu(self.upconv5(torch.cat((x, x5), 1)))
        x = self.relu(self.upconv4(torch.cat((x,x4),1)))
        x = self.relu(self.upconv3(torch.cat((x,x3),1)))
        x = self.relu(self.upconv2(torch.cat((x,x2),1)))
        x = self.up(x)

        x = self.relu(self.conv_final(torch.cat((x, x1), 1)))
        x = self.out(x)
        x = self.relu(x)
        x = self.resnet1(x)
        x = self.resnet2(x)

        x = torch.clamp(x, 0, 255)
        '''

        '''
        # Real U-Net with convolution instead of sampling 
        # Encoder
        x1 = self.down_conv_1(test_input) #
        x2 = self.down_sample_1(x1)
        x3 = self.down_conv_2(x2) #
        x4 = self.down_sample_2(x3)
        x5 = self.down_conv_3(x4) #
        x6 = self.down_sample_3(x5)
        x7 = self.down_conv_4(x6) #
        x8 = self.down_sample_4(x7)
        x9 = self.down_conv_5(x8)

        # Decoder
        x = self.up_sample_0(x9)
        x = self.up_conv_1(torch.cat([x, x7], 1))
        x = self.up_sample_1(x)
        x = self.up_conv_2(torch.cat([x, x5], 1))
        x = self.up_sample_2(x)
        x = self.up_conv_3(torch.cat([x, x3], 1))
        x = self.up_sample_3(x)
        x = self.up_conv_4(torch.cat([x, x1], 1))

        x = self.resnet1(x)
        x = self.resnet2(x)

        x = self.out(x)

        x = torch.clamp(x, 0, 255)
        '''

        '''
        # U-net with addition instead of concatenation
        # Encoder
        x1 = self.down_conv_1(test_input)  #
        x2 = self.down_sample_1(x1)
        x3 = self.down_conv_2(x2)  #
        x4 = self.down_sample_2(x3)
        x5 = self.down_conv_3(x4)  #
        x6 = self.down_sample_3(x5)
        x7 = self.down_conv_4(x6)  #
        x8 = self.down_sample_4(x7)
        x9 = self.down_conv_5(x8)

        # Decoder
        x = self.up_sample_0(x9)
        x = self.up_conv_1(x+x7)
        x = self.up_sample_1(x)
        x = self.up_conv_2(x+x5)
        x = self.up_sample_2(x)
        x = self.up_conv_3(x+x3)
        x = self.up_sample_3(x)
        x = self.up_conv_4(x+x1)

        x = self.out(x)

        x = torch.clamp(x, 0, 255)
        '''

        '''
        # Modification on the previous one
        x1 = self.down_conv_1(test_input)  #
        #x1_r = self.resnet1(x1)
        x2 = self.down_sample_1(x1)
        x3 = self.down_conv_2(x2)  #
        #x3_r = self.resnet2(x3)
        x4 = self.down_sample_2(x3)
        x5 = self.down_conv_3(x4)  #
        #x5_r = self.resnet3(x5)
        x6 = self.down_sample_3(x5)
        x7 = self.down_conv_4(x6)  #
        #x7_r = self.resnet4(x7)
        x8 = self.down_sample_4(x7)
        x9 = self.down_conv_5(x8)

        # Decoder
        x = self.up_sample_0(x9)
        x = self.up_conv_1(x + x7)
        #x = self.resnet5(x)
        x = self.up_sample_1(x)
        x = self.up_conv_2(x + x5)
        #x = self.resnet6(x)
        x = self.up_sample_2(x)
        x = self.up_conv_3(x + x3)
        #x = self.resnet7(x)
        x = self.up_sample_3(x)
        x = self.up_conv_4(x + x1)

        #x = self.resnet8(x)
        x = self.out(x)

        x = torch.clamp(x, 0, 255)
        '''


        # U-net with normalization
        # Encoder
        x1 = self.down_conv_1(test_input)  #
        x2 = self.down_sample_1(x1)
        x3 = self.down_conv_2(x2)  #

        #x4 = self.down_sample_2(x3)
        #x5 = self.down_conv_3(x4)  #

        #x6 = self.down_sample_3(x5)
        #x7 = self.down_conv_4(x6)  #
        #x8 = self.down_sample_4(x7)
        #x9 = self.down_conv_5(x8)
        #x10 = self.down_sample_5(x9)
        #x11 = self.down_conv_6(x10)

        # Decoder
        #x = self.up_conv_0(x11)
        #x = self.up_sample_0(x)
        #x = self.up_conv_1(torch.cat([x, x9], 1))
        #x = self.up_sample_1(x)
        #x = self.up_conv_2(x7)
        #x = self.up_sample_2(x)

        #x = self.up_conv_3(torch.cat([x,x3],1))
        #x = self.up_sample_3(x)
        x = self.up_conv_4(x3)
        x = self.up_sample_4(x)
        x = self.up_conv_5(torch.cat([x,x1],1))


        #x = self.resnet1(x)
        #x = self.resnet2(x)

        x = self.out(x)


        x = torch.clamp(x, 0, 255)


        return x
