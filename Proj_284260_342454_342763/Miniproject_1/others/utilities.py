import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.nn.modules.utils import _pair, _quadruple
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader


def psnr(denoised, ground_truth):
    # Peak Signal to Noise Ratio : denoised and ground˙truth have range [0 , 1]
    mse = torch . mean ((denoised - ground_truth ) ** 2)
    return -10 * torch . log10(mse/(255**2) + 10**-8)


class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x



def double_conv(in_c,out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c,out_c,kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

    return conv



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
        y = F.relu(y)
        y = self.conv2(y)
        if self.batch_normalization: y = self.bn2(y)
        if self.skip_connections: y = y + x
        y = F.relu(y)

        return y


class MyDataset(Dataset):
    def __init__(self, input, target, transform=None):
        self.input = input
        self.target = target
        self.transform = transform

    def __getitem__(self, index):
        train_input = self.input[index]
        train_target = self.target[index]
        if self.transform is not None:
            train_input = torch.cat([self.input, self.transform(self.input)],0)
            train_target = torch.cat([self.target, self.transform(self.target)])

        return train_input, train_target

    def __len__(self):
        if self.transform is not None:
            return 3*self.input.size(0)

        return self.input.size(0)



