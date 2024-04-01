import os
import numpy as np
from torchvision.io import read_image
import fnmatch
from torch.utils.data import Dataset
import torch.nn as nn
from PIL import Image
from scipy.ndimage import uniform_filter
import random
import torch


# class CustomImageDataset(Dataset):
#     def __init__(self, img_dir, transform=None):
#         self.img_dir = img_dir
#         self.transform = transform
#         self.LEN = len( fnmatch.filter(os.listdir(img_dir), '*.png') )
#         print(self.LEN)
#
#     def __len__(self):
#         return self.LEN
#
#     def __getitem__(self, idx):
#         # print(idx)
#         img_path = os.path.join(self.img_dir, '0_0000' + str(idx) + '.png')
#         image = Image.open(img_path).convert('L')
#         if self.transform:
#             image = self.transform(image)
#         return (image, '')

class CustomImageDataset(Dataset):

  def __init__(self, img_dir, transform=None):
   self.img_dir = img_dir
   self.transform = transform
   self.filenames = os.listdir(img_dir)

  def __len__(self):
   return len(self.filenames)

  def __getitem__(self, idx):
   filename = self.filenames[idx]
   img_path = os.path.join(self.img_dir, filename)

   image = Image.open(img_path).convert('L')

   if self.transform:
     image = self.transform(image)

   return (image, '')


# 实现了对图像数据添加高斯噪声的功能
class NoisyDataset(nn.Module):
  def __init__(self, rootdir='./', mean = 0, var = 1):
    super(NoisyDataset, self).__init__()
    self.mean = mean
    self.var = var

  def forward(self, image):
    if image.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.

    noise = np.random.normal(self.mean, self.var, size = image.shape)
    noisy_image = np.clip(image + noise, low_clip, 1)
    return noisy_image

# 斑点噪声模型 u(x) = v(x) + sqrt(v(x)) * y(x),其中 y(x) 是均值为 0，方差为 δ**2 的高斯噪声。
class SpeckleNoisyDataset(nn.Module):
  def __init__(self, rootdir='./', sigma=0.5):
    super(SpeckleNoisyDataset, self).__init__()
    self.sigma = sigma

  def forward(self, image):
    if image.min() < 0:
      low_clip = -1.
    else:
      low_clip = 0.
    y = np.random.normal(0, self.sigma, size=image.shape)  # 高斯噪声
    noisy_image = np.clip(image + np.sqrt(np.abs(image)) * y, low_clip, 1)
    return noisy_image

# class SpeckleNoisyDataset(nn.Module):
#     def __init__(self, rootdir='./', sigmas=[0.1, 0.2, 0.4, 0.6, 0.8], mode='train'):
#         super(SpeckleNoisyDataset, self).__init__()
#         self.sigmas = sigmas
#         self.mode = mode
#         self.current_sigma = sigmas[0]  # 默认使用第一个sigma值
#
#     def set_mode(self, mode):
#         """切换训练模式和测试模式"""
#         assert mode in ['train', 'test'], "Mode must be 'train' or 'test'."
#         self.mode = mode
#
#     def set_sigma(self, sigma):
#         """测试模式下设置固定的sigma值"""
#         self.current_sigma = sigma
#
#     def forward(self, image):
#         # 确定图像的最小值，以便确定低剪辑阈值
#         if image.min() < 0:
#             low_clip = -1.
#         else:
#             low_clip = 0.
#         # 训练模式下随机选择一个sigma值，测试模式下使用固定的sigma值
#         sigma = random.choice(self.sigmas) if self.mode == 'train' else self.current_sigma
#
#         # 确保图像为浮点类型
#         # image = np.array(image) / 255.0
#         # print(image)
#         # 添加均值为0，方差为sigma^2 的乘性噪声
#         nn = sigma * np.random.randn(*image.shape)+0
#         # 对每个图像的每个通道应用3x3均值滤波
#         #一个批次的图像数据（一个四维数组），
#         # 但在调用 uniform_filter 时，只提供了一个单一的滤波大小（(3, 3)），这适用于单个二维图像
#         # 考虑为每个通道单独应用uniform_filter
#         for i in range(nn.shape[0]):  # 批次大小
#             for j in range(nn.shape[1]):  # 通道数
#                 nn[i, j] = uniform_filter(nn[i, j], size=3)
#         # 3x3均值滤波
#         # nn = uniform_filter(nn, size=(3, 3))
#         # 将滤波后的噪声添加到原图像上
#         # noisy_image = image + np.sqrt(image) * nn
#         # 图像强度值规范化到[0, 255]并转换为uint8
#         # noisy_image = np.clip(noisy_image * 255, 0, 255).astype(np.float32)
#         noisy_image = np.clip(image + np.sqrt(np.abs(image)) * nn, low_clip, 1)
#         # noisy_image = torch.from_numpy(noisy_image).unsqueeze(0)
#         return noisy_image



import numpy as np
import torch.nn as nn
# 实现了对图像数据添加斑点噪声的功能 斑点噪声常被认为是由图像中的每个像素乘以一个随机噪声值产生的。
# 假设斑点噪声服从均值为 mean ，方差为 var 的正态分布。并且，我们使用乘性噪声模型，即将图像的像素值乘以噪声值，来模拟斑点噪声。
# class NoisyDataset(nn.Module):
#   def __init__(self, rootdir='./', mean=0, var=0.1):
#     super(NoisyDataset, self).__init__()
#     self.mean = mean
#     self.var = var
#
#   def forward(self, image):
#     if image.min() < 0:
#       low_clip = -1.
#     else:
#       low_clip = 0.
#     noise = np.random.normal(self.mean, self.var ** 0.5, size = image.shape)
#     noisy_image = np.clip(image * noise, low_clip, 1)
#     return noisy_image


# class NoisyDataset(Dataset):
#   def __init__(self, rootdir):
#       super().__init__()
#       self.img_dir = rootdir
#       self.images = read_noisy_images(rootdir)
#
#   def forward(self, idx):
#       image = self.images[idx]
#       return image
#
# def read_noisy_images(rootdir):
#   images = []
#   for filename in os.listdir(rootdir):
#       img = Image.open(os.path.join(rootdir, filename)).convert('L')
#   images.append(np.array(img))
#   return images

