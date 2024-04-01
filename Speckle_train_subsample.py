import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

from dataset import CustomImageDataset, SpeckleNoisyDataset
from UNet import UNet
from utils import imshow
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
import pytorch_ssim
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
import os

os.environ["CUDA_DEVICES_ORDER"]="PCI_BUS_IS"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Neighbour2Neighbour():
    def __init__(self, gamma=1, k=2, alpha = 1):
        self.gamma = gamma
        self.k = k
        self.alpha = alpha
        self.EPOCHS, self.BATCH, self.sigma, self.LR, self.DATA_DIR, self.CH_DIR = self.__get_args__()
        self.transforms = transforms.Compose(
            [transforms.CenterCrop(256),
             transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))])
        self.trainloader, self.validloader = self.load_data()
        self.use_cuda = torch.cuda.is_available()

    def __get_args__(self):
        parser = argparse.ArgumentParser(description='Parameters')
        parser.add_argument('--epochs', type=int, default=30)
        parser.add_argument('--batch', type=int, default=4)
        parser.add_argument('--sigma', type=float, default=0.5)
        parser.add_argument('--learning_rate', type=float, default=.0005)
        parser.add_argument('--data_dir', type=str, default='./data')
        parser.add_argument('--checkpoint_dir', type=str,
                            default='./checkpoints')

        args = parser.parse_args()
        return (args.epochs, args.batch, args.sigma, args.learning_rate, args.data_dir, args.checkpoint_dir)

    def subsample(self, image):
        blen, channels, m, n = np.shape(image)
        dim1, dim2 = m // self.k, n // self.k
        images = [np.zeros([blen, channels, dim1, dim2]) for _ in range(4)]
        image_cpu = image.cpu()

        for channel in range(channels):
            for i in range(dim1):
                for j in range(dim2):
                    i1, j1 = i * self.k, j * self.k
                    # 将像素分配到四个子图像中
                    images[0][:, channel, i, j] = image_cpu[:, channel, i1, j1]  # 左上角
                    images[1][:, channel, i, j] = image_cpu[:, channel, i1, j1 + 1]  # 右上角
                    images[2][:, channel, i, j] = image_cpu[:, channel, i1 + 1, j1]  # 左下角
                    images[3][:, channel, i, j] = image_cpu[:, channel, i1 + 1, j1 + 1]  # 右下角

        if self.use_cuda:
            return [torch.from_numpy(img).cuda() for img in images]
        return [torch.from_numpy(img) for img in images]


    def load_data(self):
        trainset = CustomImageDataset(
            self.DATA_DIR + '/train/', transform=self.transforms)
        validset = CustomImageDataset(
            self.DATA_DIR + '/val/', transform=self.transforms)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.BATCH, num_workers=2, shuffle=True,)
        validloader = torch.utils.data.DataLoader(
            validset, batch_size=self.BATCH, num_workers=2, shuffle=True,)
        return trainloader, validloader

    def get_model(self):
        model = UNet(in_channels=1, out_channels=1).double()
        if self.use_cuda:
            model = model.cuda()
        noisy = SpeckleNoisyDataset(sigma=self.sigma)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.LR)
        criterion = RegularizedLoss()
        return model, noisy, optimizer, criterion

    def train(self):
        model, noisy, optimizer, criterion = self.get_model()
        psnr_epochs = []
        ssim_epochs = []
        if self.use_cuda:
            model = model.cuda()

        min_loss_valid = 100
        for epoch in range(self.EPOCHS):
            total_loss_valid = 0
            total_loss = 0
            for idx, (batch, _) in enumerate(self.trainloader):
                optimizer.zero_grad()
                noisy_image = noisy(batch)
                # noisy_image = DataLoader(batch)
                if self.use_cuda:
                    noisy_image = noisy_image.cuda()
                g1, g2, g3, g4 = self.subsample(noisy_image)
                fg1 = model(g1)
                fg3 = model(g3)
                with torch.no_grad():
                    X = model(noisy_image)
                    G1, G2, G3, G4 = self.subsample(X)
                total_loss1 = criterion(g1, g2, fg1, G1, G2)
                # total_loss = criterion(fg1, g2, G1, G2)
                total_loss2 = criterion(g3, g4, fg3, G3, G4)
                total_loss = (total_loss1+total_loss2)/2
                total_loss.backward()
                optimizer.step()

            for idx, (batch, _) in enumerate(self.validloader):
                with torch.no_grad():
                    noisy_image = noisy(batch)
                    if self.use_cuda:
                        noisy_image = noisy_image.cuda()
                    g1, g2, g3, g4 = self.subsample(noisy_image)
                    fg1 = model(g1)
                    fg3 = model(g3)
                    X = model(noisy_image)
                    G1, G2, G3, G4 = self.subsample(X)
                    total_loss_valid1 = criterion(g1, g2, fg1, G1, G2)
                    total_loss_valid2 = criterion(g3, g4, fg3, G3, G4)
                    total_loss_valid = (total_loss_valid1+total_loss_valid2)/2
                    # total_loss_valid = criterion(fg1, g2, G1, G2)

            if total_loss_valid < min_loss_valid:
                min_loss_valid = total_loss_valid

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, self.CH_DIR + '/chk_speckle_subsample_SSIM_epochs30' + str(self.k) + '_' + str(self.gamma) + '_' + str(self.alpha)+'_'+str(self.sigma)+'.pt')
                # }, self.CH_DIR + '/chk_' + str(self.k) + '_' + str(self.gamma) + '_' + str(self.sigma) + '.pt')
                print('Saving Model...')
            print('Epoch', epoch+1, 'Loss Valid:',
                  total_loss_valid, 'Train', total_loss)

            # 在每个epoch结束时计算验证集上的PSNR和SSIM
            psnr_epoch, ssim_epoch = self.calculate_metrics_on_validation_set()
            psnr_epochs.append(psnr_epoch)
            ssim_epochs.append(ssim_epoch)
            # 绘制折线图
        self.plot_metrics(psnr_epochs, ssim_epochs)

    def calculate_metrics_on_validation_set(self, model, validloader):
        total_psnr = 0.0
        total_ssim = 0.0
        num_batches = 0

        # 评估模式
        model.eval()

        with torch.no_grad():
            for batch, _ in validloader:
                # noisy_image = self.noisy(batch)  # 假设您已经有了添加噪声的逻辑
                noisy_image = SpeckleNoisyDataset(sigma=self.sigma)
                if self.use_cuda:
                    noisy_image = noisy_image.cuda()
                    batch = batch.cuda()
                clean_image = model(noisy_image)  # 假设模型输出的是去噪后的图像

                # 计算每个batch的PSNR和SSIM并累加
                total_psnr += psnr(batch, clean_image).item()
                total_ssim += calculate_ssim(batch, clean_image).item()
                num_batches += 1

        # 计算并返回每个epoch的平均PSNR和SSIM
        avg_psnr = total_psnr / num_batches
        avg_ssim = total_ssim / num_batches

        # 训练模式恢复
        model.train()

        return avg_psnr, avg_ssim

    def plot_metrics(self, psnr_epochs, ssim_epochs):
        epochs = range(1, self.EPOCHS + 1)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, psnr_epochs, marker='o', color='b')
        plt.title('PSNR vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')

        plt.subplot(1, 2, 2)
        plt.plot(epochs, ssim_epochs, marker='o', color='r')
        plt.title('SSIM vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('SSIM')

        plt.tight_layout()
        plt.show()


class RegularizedLoss(nn.Module):
    def __init__(self, gamma=1, alpha = 0.5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        # self.ssim_module = pytorch_ssim.SSIM(window_size=11)

    def mseloss(self, image, target):
        x = ((image - target) ** 2)
        return torch.mean(x)

    def regloss(self, g1, g2, G1, G2):
        return torch.mean((g1 - g2 - G1 + G2) ** 2)

    # def ssimloss(self, g1, g2):
    #     # Compute SSIM loss on GPU
    #     return 1 - self.ssim_module(g1, g2)

    def ssimloss(self, g1, g2):
        g1 = g1.cpu().detach().numpy()
        g2 = g2.cpu().detach().numpy()
        return 1 - ssim(g1, g2, data_range=1, win_size=3, channel_axis=True, padding=True)
        #ssim 函数默认假定输入是多通道图像,图像是单通道的，将 multichannel 参数设置为 False
    # def mssimloss(self, g1, g2):
    #     # 计算 MSSIM 损失
    #     return 1 - ms_ssim(g1, g2, data_range=1, size_average=True)
    # def sim_loss(self, g1, g2):
    #     cos_sim = F.cosine_similarity(g1, g2, dim=1)
    #     return (1 - cos_sim).mean()

    def forward(self, g1, g2, fg1, G1f, G2f):
        mseloss = self.mseloss(fg1, g2)

        # Compute the SSIM loss
        # Compute the SSIM loss
        ssimloss = self.ssimloss(g1, g2)
        # cosloss = self.sim_loss(g1, g2)
        # mssimloss = self.mssimloss(g1, g2)

        regloss = self.regloss(fg1, g2, G1f, G2f)

        return mseloss + self.gamma * regloss + self.alpha * ssimloss

if __name__ == '__main__':
    N2N = Neighbour2Neighbour(gamma=1)
    N2N.train()
