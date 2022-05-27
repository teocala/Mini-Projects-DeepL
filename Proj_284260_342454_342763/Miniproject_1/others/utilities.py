import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn



def compute_psnr(x, y, max_range=255.0):
    assert x.shape == y.shape and x.ndim == 4
    return 20 * torch.log10(torch.tensor(max_range)) - 10 * torch.log10(((x-y) ** 2).mean((1,2,3))).mean()


class ResNetBlock(nn.Module):
    def __init__(self, nb_channels, kernel_size, batch_normalization=True, skip_connections=True):
        super().__init__()

        self.batch_normalization = batch_normalization
        self.skip_connections = skip_connections

        self.conv1 = nn.Conv2d(nb_channels, nb_channels,
                               kernel_size = kernel_size,
                               padding = (kernel_size - 1) // 2)

        self.bn1 = nn.BatchNorm2d(nb_channels)

        self.conv2 = nn.Conv2d(nb_channels, nb_channels,
                               kernel_size = kernel_size,
                               padding = (kernel_size - 1) // 2)

        self.bn2 = nn.BatchNorm2d(nb_channels)

    def forward(self, x):
        y = self.conv1(x)
        if self.batch_normalization : y = self.bn1(y)
        y = F.leaky_relu(y)
        y = self.conv2(y)
        if self.batch_normalization: y = self.bn2(y)
        if self.skip_connections: y = y + x
        y = F.leaky_relu(y)

        return y


class ResNetBlock2(nn.Module):
    def __init__(self, nb_channels, kernel_size, batch_normalization=True, skip_connections=True):
        super().__init__()

        self.batch_normalization = batch_normalization
        self.skip_connections = skip_connections

        self.conv1 = nn.Conv2d(nb_channels, nb_channels,
                               kernel_size = kernel_size,
                               padding = (kernel_size - 1) // 2)

        self.bn1 = nn.GroupNorm(num_groups=3,num_channels=nb_channels)

        self.conv2 = nn.Conv2d(nb_channels, nb_channels,
                               kernel_size = kernel_size,
                               padding = (kernel_size - 1) // 2)

        self.bn2 = nn.GroupNorm(num_groups=3,num_channels=nb_channels)

    def forward(self, x):
        y = self.conv1(x)
        if self.batch_normalization : y = self.bn1(y)
        y = F.leaky_relu(y)
        y = self.conv2(y)
        if self.batch_normalization: y = self.bn2(y)
        if self.skip_connections: y = y + x
        y = F.leaky_relu(y)

        return y



class MyData(Dataset):
    def __init__(self, train_input, train_target):
        self.train_input = train_input
        self.train_target = train_target
        self.transform1 = transforms.RandomHorizontalFlip(p=1)
        self.transform2 = transforms.RandomVerticalFlip(p=1)

    def __getitem__(self, index):
        return self.train_input[index], self.transform1(self.train_input[index]), self.transform2(self.train_input[index]), \
               self.train_target[index], self.transform1(self.train_target[index]), self.transform2(self.train_target[index])

    def __len__(self):
        return self.train_input.size(0)


def collate_batch(batch):
    train_list = []
    target_list = []
    for item1,item2,item3, target1,target2,target3 in batch:
        train_list.extend([item1,item2,item3])
        target_list.extend([target1,target2,target3])

    return torch.stack(train_list), torch.stack(target_list)


class UNet(nn.Module):
    """U-Net Architecture"""

    def __init__(self, in_channels=3, out_channels=3):
        """Initializes U-Net."""

        super(UNet, self).__init__()

        self._block1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(48,48,kernel_size=3,stride=2,padding=1))


        self._block2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(48,48,kernel_size=3,padding=1,stride=2))


        self._block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))


        self._block4 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))


        self._block5 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))


        self._block6 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
            )

        # Initialize weights
        self._init_weights()


    def _init_weights(self):
        """Initializes weights using Xavier initialization"""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()


    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """

        # Encoder
        down1 = self._block1(x)
        down2 = self._block2(down1)
        down3 = self._block2(down2)
        down4 = self._block2(down3)
        down5 = self._block2(down4)

        # Decoder
        up5 = self._block3(down5)
        concat5 = torch.cat((up5, down4), dim=1)
        up4 = self._block4(concat5)
        concat4 = torch.cat((up4, down3), dim=1)
        up3 = self._block5(concat4)
        concat3 = torch.cat((up3, down2), dim=1)
        up2 = self._block5(concat3)
        concat2 = torch.cat((up2, down1), dim=1)
        up1 = self._block5(concat2)
        concat1 = torch.cat((up1, x), dim=1)

        # Final activation
        return self._block6(concat1)



class ResUNet(nn.Module):
    """Residual U-Net Architecture"""

    def __init__(self, in_channels=3, out_channels=3):
        """Initializes Residual U-Net"""

        super(ResUNet, self).__init__()

        self.in_ = nn.Conv2d(in_channels,48,kernel_size=1,stride=1)

        self._block1 = nn.Sequential(
            ResNetBlock2(nb_channels=48,kernel_size=3,batch_normalization=True,skip_connections=True),
            nn.MaxPool2d(2))

        self._block2 = nn.Sequential(
            ResNetBlock2(nb_channels=48,kernel_size=3,batch_normalization=True,skip_connections=True),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))

        self._block3 = nn.Sequential(
            ResNetBlock2(nb_channels=96,kernel_size=3,batch_normalization=True,skip_connections=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))


        self._block4 = nn.Sequential(
            nn.Conv2d(144, 96, 1, stride=1),
            nn.LeakyReLU(0.1),
            ResNetBlock2(nb_channels=96,kernel_size=3,batch_normalization=True,skip_connections=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))


        self._block5 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
            )

        # Initialize weights
        self._init_weights()


    def _init_weights(self):
        """Initializes weights using Xavier initizialitation"""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()


    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """

        # Encoder
        x_in = self.in_(x)
        down1 = self._block1(x_in)
        down2 = self._block1(down1)
        down3 = self._block1(down2)
        down4 = self._block1(down3)
        down5 = self._block1(down4)

        # Decoder
        up5 = self._block2(down5)
        concat5 = torch.cat((up5, down4), dim=1)
        up4 = self._block3(concat5)
        concat4 = torch.cat((up4, down3), dim=1)
        up3 = self._block4(concat4)
        concat3 = torch.cat((up3, down2), dim=1)
        up2 = self._block4(concat3)
        concat2 = torch.cat((up2, down1), dim=1)
        up1 = self._block4(concat2)
        concat1 = torch.cat((up1, x), dim=1)

        # Final activation
        return self._block5(concat1)

















