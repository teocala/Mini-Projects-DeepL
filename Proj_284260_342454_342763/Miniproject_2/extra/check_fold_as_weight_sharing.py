# -*- coding: utf-8 -*-
"""
Created on Sat May 21 10:51:26 2022

@author: matte
"""
import torch
from torch.nn.functional import fold, unfold

A = torch.arange(5*6*7*8, dtype = float)
A = A.view([5,6,7,8])


unfolded = unfold(A, kernel_size=[5,5], stride=2)
folded = fold(unfolded, A.shape[2:], kernel_size=[5,5], stride=2)


count_tensor = torch.zeros([5*6*7*8])
for i in range(5*6*7*8):
    count_tensor[i] = torch.sum(unfolded==i)
count_tensor = count_tensor.view(A.shape)




print (torch.equal(folded, count_tensor.multiply(A)))

