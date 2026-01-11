import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import json
import random
import numpy as np
import itertools


class ContrastDataset_eval(Dataset):
    def __init__(self, data_dir,noise_dir,class1,class2, transform=None,real = True):
        """
        初始化数据集，读取数据目录和数据变换。

        Args:
            data_dir (str): 数据目录。
            transform (callable, optional): 应用于数据的变换。
        """
        self.data = os.listdir(os.path.join(data_dir,'jpg50','sdv21'))
        self.data_dir = data_dir
        self.real = real
        self.class1, self.class2 = class1,class2

        if real:
            self.pairs = np.array([
                [0, 1], [0, 2], [0, 3],
                [1, 2], [1, 3],
                [2, 3],
            ])
            self.pairs_labels = np.array([0,0,0,1,1,1])
        else:
            self.pairs = np.array([
                [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
                [1, 2], [1, 3], [1, 4], [1, 5],
                [2, 3], [2, 4], [2, 5],
                [3, 4], [3, 5],
                [4, 5]
            ])
            self.pairs_labels = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1])

        # img_list = [i for i in range(10)]
        # self.pairs = np.array(list(itertools.combinations(img_list, 2)))
        # labels = [0,0,0,1,1,1,2,2,2,3]
        #
        # results = []
        # for combo in self.pairs:
        #     idx1 = img_list.index(combo[0])
        #     idx2 = img_list.index(combo[1])
        #     if labels[idx1] == labels[idx2]:
        #         results.append(1)
        #     else:
        #         results.append(0)
        #
        # self.pairs_labels = np.array(results)

    def get_images(self,):

        pass

    def __len__(self):
        """
        返回数据集的大小。
        """
        return len(self.data) // 3

    def __getitem__(self, idx):
        """
        获取索引对应的数据及标签。

        Args:
            idx (int): 数据索引。

        Returns:
            images (tensor): 图像
            labels (tensor): 所属模型标签
        """
        # images_list = ([os.path.join(self.data_dir,'lcm',str(idx) + '_' + str(i) + '.npy') for i in range(3)] +
        #                [os.path.join(self.data_dir, 'sdv14', str(idx) + '_' + str(i) + '.npy') for i in range(3)])
                       # + [os.path.join(self.data_dir, 'ldm', str(idx) + '_' + str(i) + '.npy') for i in range(3)]
                       # + [os.path.join(self.data_dir, 'real', str(idx) + '_0.npy')])

        if self.real:
            images_list = ([os.path.join(self.data_dir, 'real', str(idx) + '_0.npy')] +
                           [os.path.join(self.data_dir, self.class1, str(idx) + '_' + str(i) + '.npy') for i in range(3)])
        else:
            images_list = (
                        [os.path.join(self.data_dir, self.class1, str(idx) + '_' + str(i) + '.npy') for i in range(3)] +
                        [os.path.join(self.data_dir, self.class2, str(idx) + '_' + str(i) + '.npy') for i in range(3)])
        images = []

        for item in images_list:
            img = np.load(item)
            images.append(torch.from_numpy(img).type(torch.float32))

        indices = np.arange(len(self.pairs))
        np.random.shuffle(indices)
        shuffled_data = self.pairs[indices]
        labels = self.pairs_labels[indices]

        x1 = []
        x2 = []

        for x,y in shuffled_data:
            x1.append(images[x].view(1,-1))
            x2.append(images[y].view(1,-1))



        return torch.cat(x1),torch.cat(x2),torch.from_numpy(labels).type(torch.float32)


class ContrastDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        初始化数据集，读取数据目录和数据变换。

        Args:
            data_dir (str): 数据目录。
            transform (callable, optional): 应用于数据的变换。
        """
        self.data = os.listdir(os.path.join(data_dir,'sdv21'))
        self.data_dir = data_dir
        self.style_feature_dir = '/home/jxq/Datasets/style_clip'

        self.text_feature_dir = '/home/jxq/Datasets/clip_encoder_feature/text_features'

        # self.pairs = np.array([
        #     [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
        #     [1, 2], [1, 3], [1, 4], [1, 5],
        #     [2, 3], [2, 4], [2, 5],
        #     [3, 4], [3, 5],
        #     [4, 5]
        # ])
        # self.pairs_labels = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1])

        self.pairs = np.array([
            [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8],
            [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8],
            [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8],
            [3, 4], [3, 5], [3, 6], [3, 7], [3, 8],
            [4, 5], [4, 6], [4, 7], [4, 8],
            [5, 6], [5, 7], [5, 8],
            [6, 7], [6, 8],
            [7, 8],
        ])
        self.pairs_labels = np.array([1, 1, 0, 0, 0, 0, 0, 0,
                                      1, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0,
                                      1, 1, 0, 0, 0,
                                      1, 0, 0, 0,
                                      0, 0, 0,
                                      1, 1,
                                      1])


    def get_images(self,):

        pass

    def __len__(self):
        """
        返回数据集的大小。
        """
        return len(self.data) // 3

    def __getitem__(self, idx):
        """
        获取索引对应的数据及标签。

        Args:
            idx (int): 数据索引。

        Returns:
            images (tensor): 图像
            labels (tensor): 所属模型标签
        """
        images_list = ([os.path.join(self.style_feature_dir,'lcm',str(idx) + '_' + str(i) + '.npy') for i in range(3)] +
                       [os.path.join(self.style_feature_dir, 'sdv21', str(idx) + '_' + str(i) + '.npy') for i in range(3)]
                       + [os.path.join(self.style_feature_dir, 'sdv14', str(idx) + '_' + str(i) + '.npy') for i in range(3)])

        clip_feature_list = ([os.path.join(self.data_dir,'lcm',str(idx) + '_' + str(i) + '.npy') for i in range(3)] +
                       [os.path.join(self.data_dir, 'sdv21', str(idx) + '_' + str(i) + '.npy') for i in range(3)]
                       + [os.path.join(self.data_dir, 'sdv14', str(idx) + '_' + str(i) + '.npy') for i in range(3)])

        text_featrue = np.load(os.path.join(self.text_feature_dir,str(idx)+ '.npy'))

        images1 = []

        for item in images_list:
            img = np.load(item)
            images1.append(torch.from_numpy(img).type(torch.float32))

        indices = np.arange(len(self.pairs))
        np.random.shuffle(indices)
        shuffled_data = self.pairs[indices]
        labels = self.pairs_labels[indices]

        x1 = []
        x2 = []

        for x,y in shuffled_data:
            x1.append(images1[x].view(1,-1))
            x2.append(images1[y].view(1,-1))

        images = []

        for item in clip_feature_list:
            img = np.load(item)
            images.append(torch.from_numpy(img).type(torch.float32))

        return torch.cat(x1),torch.cat(x2),torch.cat(images),torch.from_numpy(labels).type(torch.float32), torch.from_numpy(text_featrue).type(torch.float32)

def collate_fn(batch):
    images1, images2, images, labels, text_featrue = zip(*batch)
    # 将images展开为(batch_size*7, 3, image_size, image_size)的形状
    images1 = torch.cat(images1)
    images2 = torch.cat(images2)
    images = torch.cat(images)
    labels = torch.cat(labels)
    return images1, images2, images, labels.view(-1,1), torch.cat(text_featrue)

def collate_fn1(batch):
    images1, images2, labels = zip(*batch)
    # 将images展开为(batch_size*7, 3, image_size, image_size)的形状
    images1 = torch.cat(images1)
    images2 = torch.cat(images2)
    labels = torch.cat(labels)
    return images1, images2, labels.view(-1, 1)

if __name__ == '__main__':
    # 定义数据变换
    from torchvision import transforms

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 创建数据集实例
    data_dir = '/home/jxq/Datasets/clip_encoder_feature'
    dataset = ContrastDataset(data_dir)

    # 创建 DataLoader 实例
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_fn)

    # 使用 DataLoader 进行训练迭代
    for images1, images2, images, labels, text_featrue in dataloader:
        print(images1.shape)  # (batch_size*7, 3, image_size, image_size)
        print(1)
