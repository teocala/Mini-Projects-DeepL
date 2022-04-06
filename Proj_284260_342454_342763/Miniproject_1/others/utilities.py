import torch

def psnr(denoised, ground_truth):
    # Peak Signal to Noise Ratio : denoised and ground˙truth have range [0 , 1]
    mse = torch . mean ((denoised - ground_truth ) ** 2)
    return -10 * torch . log10 ( mse + 10** -8)