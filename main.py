import logging
import time
import os
import torch
from torch import optim
from torch.utils.data import DataLoader,SubsetRandomSampler
from tqdm import tqdm
import torch.utils.data.distributed
import torch.distributed as dist
import numpy as np
import random
import argparse
from torch.utils.tensorboard import SummaryWriter
from dataset_loader import ContrastDataset,collate_fn
from torchvision import transforms
import torch.nn as nn
from tools import AverageMeter
from style_clip_model import SiameseNetwork, augment_net
import torch.nn.functional as F
from contrastive_loss import SupContrastiveloss
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 设置随机数种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(4396)  # 设定一个任意的种子值

pairs = np.array([
[9, 0], [9, 1], [9, 2], [9, 3], [9, 4], [9, 5], [9, 6], [9, 7], [9, 8], [9, 10], [9, 11], [9, 12], [9, 13], [9, 14], [9, 15], [9, 16], [9, 17],
[10, 0], [10, 1], [10, 2], [10, 3], [10, 4], [10, 5], [10, 6], [10, 7], [10, 8], [10, 11], [10, 12], [10, 13], [10, 14], [10, 15], [10, 16], [10, 17],
[11, 0], [11, 1], [11, 2], [11, 3], [11, 4], [11, 5], [11, 6], [11, 7], [11, 8], [11, 12], [11, 13], [11, 14], [11, 15], [11, 16], [11, 17],
[12, 0], [12, 1], [12, 2], [12, 3], [12, 4], [12, 5], [12, 6], [12, 7], [12, 8], [12, 13], [12, 14], [12, 15], [12, 16], [12, 17],
[13, 0], [13, 1], [13, 2], [13, 3], [13, 4], [13, 5], [13, 6], [13, 7], [13, 8], [13, 14], [13, 15], [13, 16], [13, 17],
[14, 0], [14, 1], [14, 2], [14, 3], [14, 4], [14, 5], [14, 6], [14, 7], [14, 8], [14, 15], [14, 16], [14, 17],
[15, 0], [15, 1], [15, 2], [15, 3], [15, 4], [15, 5], [15, 6], [15, 7], [15, 8], [15, 16], [15, 17],
[16, 0], [16, 1], [16, 2], [16, 3], [16, 4], [16, 5], [16, 6], [16, 7], [16, 8], [16, 17],
[17, 0], [17, 1], [17, 2], [17, 3], [17, 4], [17, 5], [17, 6], [17, 7], [17, 8]
])

pairs_labels = np.array(
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
0, 0, 0, 0, 0, 0, 0, 0, 0
]
)

pairs2 = np.array([
[18, 0], [18, 1], [18, 2], [18, 3], [18, 4], [18, 5], [18, 6], [18, 7], [18, 8], [18, 9], [18, 10], [18, 11], [18, 12], [18, 13], [18, 14], [18, 15], [18, 16], [18, 17], [18, 19], [18, 20], [18, 21], [18, 22], [18, 23], [18, 24], [18, 25], [18, 26],
[19, 0], [19, 1], [19, 2], [19, 3], [19, 4], [19, 5], [19, 6], [19, 7], [19, 8], [19, 9], [19, 10], [19, 11], [19, 12], [19, 13], [19, 14], [19, 15], [19, 16], [19, 17], [19, 20], [19, 21], [19, 22], [19, 23], [19, 24], [19, 25], [19, 26],
[20, 0], [20, 1], [20, 2], [20, 3], [20, 4], [20, 5], [20, 6], [20, 7], [20, 8], [20, 9], [20, 10], [20, 11], [20, 12], [20, 13], [20, 14], [20, 15], [20, 16], [20, 17], [20, 21], [20, 22], [20, 23], [20, 24], [20, 25], [20, 26],
[21, 0], [21, 1], [21, 2], [21, 3], [21, 4], [21, 5], [21, 6], [21, 7], [21, 8], [21, 9], [21, 10], [21, 11], [21, 12], [21, 13], [21, 14], [21, 15], [21, 16], [21, 17], [21, 22], [21, 23], [21, 24], [21, 25], [21, 26],
[22, 0], [22, 1], [22, 2], [22, 3], [22, 4], [22, 5], [22, 6], [22, 7], [22, 8], [22, 9], [22, 10], [22, 11], [22, 12], [22, 13], [22, 14], [22, 15], [22, 16], [22, 17], [22, 23], [22, 24], [22, 25], [22, 26],
[23, 0], [23, 1], [23, 2], [23, 3], [23, 4], [23, 5], [23, 6], [23, 7], [23, 8], [23, 9], [23, 10], [23, 11], [23, 12], [23, 13], [23, 14], [23, 15], [23, 16], [23, 17], [23, 24], [23, 25], [23, 26],
[24, 0], [24, 1], [24, 2], [24, 3], [24, 4], [24, 5], [24, 6], [24, 7], [24, 8], [24, 9], [24, 10], [24, 11], [24, 12], [24, 13], [24, 14], [24, 15], [24, 16], [24, 17], [24, 25], [24, 26],
[25, 0], [25, 1], [25, 2], [25, 3], [25, 4], [25, 5], [25, 6], [25, 7], [25, 8], [25, 9], [25, 10], [25, 11], [25, 12], [25, 13], [25, 14], [25, 15], [25, 16], [25, 17], [25, 26],
[26, 0], [26, 1], [26, 2], [26, 3], [26, 4], [26, 5], [26, 6], [26, 7], [26, 8], [26, 9], [26, 10], [26, 11], [26, 12], [26, 13], [26, 14], [26, 15], [26, 16], [26, 17]])


pairs_labels2 = np.array(
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
)

data_dir = '/home/jxq/Datasets/style_clip'
data_list = os.listdir(os.path.join('/home/jxq/Datasets/style_clip/jpg75', 'real'))
class_names = ['lcm','sdv21','sdv14','lcm_aug','sdv21_aug','sdv14_aug']
fixed_palette = sns.color_palette("deep", len(class_names))
color_map = {name: color for name, color in zip(class_names, fixed_palette)}

# 定义函数以应用颜色映射
def apply_color_map(y, color_map):
    return [color_map[class_name] for class_name in y]


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def checkpoint(net, criterion, history, epoch, optimizer, count_iter, args):
    package = dict()

    package['net'] = net.state_dict()
    package['optimizer'] = optimizer.state_dict()
    package['epoch'] = epoch
    package['history'] = history
    package['iter'] = count_iter

    logging.info(f'Saving checkpoints at {epoch} epochs.')
    suffix_first = 'first.pth'
    suffix_latest = 'latest.pth'
    suffix_best = 'best.pth'

    if not os.path.exists(args.save_ckpt):
        os.makedirs(args.save_ckpt)
        torch.save(package, '{}/package_{}'.format(args.save_ckpt, suffix_latest))

    if epoch == 0:
        torch.save(net.state_dict(), '{}/package_{}'.format(args.save_ckpt, suffix_first))
        # torch.save(criterion.scale,'{}/temperature.pth'.format(args.save_ckpt))

    cur_err = package['history']['eval_loss'][-1]
    if cur_err < args.best_loss:
        args.best_loss = cur_err
        torch.save(net.state_dict(), '{}/net_{}'.format(args.save_ckpt, suffix_best))
        # torch.save(criterion.scale, '{}/temperature.pth'.format(args.save_ckpt))

def train_one_epoch(model, train_loader, optimizer, criterion, epoch, accumulated_iter, tb_log, args, augnets, augnet_criterion):
    batch_time = AverageMeter()
    dataload_time = AverageMeter()
    augnet = augment_net(1024).to(args.device)
    optimizerA = torch.optim.Adam([{'params': augnet.parameters(), 'lr': args.augnet_lr}])
    schedulerA = torch.optim.lr_scheduler.StepLR(optimizerA, step_size=args.augnet_step_size, gamma=0.9)

    num_aug_epoch = 10

    with tqdm(total=num_aug_epoch*len(train_loader), desc=f'Epoch {epoch}/{args.epochs}', unit='batch') as pbar:
        torch.cuda.synchronize()
        tic = time.perf_counter()
        for j in range(num_aug_epoch):
            train_loader_iter = iter(train_loader)
            for i, batch_data in enumerate(train_loader_iter):

                images1, images2, images, labels, text_featrues = batch_data

                images1 = images1.to(args.device)
                images2 = images2.to(args.device)
                images = images.to(args.device)
                labels = labels.to(args.device)
                text_featrues = text_featrues.to(args.device)

                torch.cuda.synchronize()
                dataload_time.update(time.perf_counter() - tic)

                # -------- step1 :train augnet start --------
                optimizerA.zero_grad()
                set_requires_grad([augnet], True)
                set_requires_grad([model], False)

                # reconstruction loss
                aug_img = augnet(images)
                # aug_img_copy = aug_img.clone()
                # images_copy = images.clone()
                # aug_img_copy = aug_img_copy / aug_img_copy.norm(dim=1, keepdim=True)
                # images_copy = images_copy / images_copy.norm(dim=1, keepdim=True)

                cosine_similarity = F.cosine_similarity(aug_img, images, dim=1)
                aug_label = torch.ones(aug_img.shape[0]).to(args.device)

                aug_loss_mse = augnet_criterion(cosine_similarity, aug_label)

                loss_A = torch.clamp(aug_loss_mse, 0)
                # losses.update({'loss_mse': loss_A.item()})
                tb_log.add_scalar('train_Aug{}/Aug{} reconstruct loss'.format(len(augnets),len(augnets)), loss_A.item(), accumulated_iter)

                # clip_style_feature_diff = augnet_criterion(aug_img, images)
                # clip_style_feature_loss = args.w_dist_clip_feature * torch.clamp(clip_style_feature_diff, 0,0.1)
                # loss_A -= clip_style_feature_loss - 1
                # tb_log.add_scalar('train_Aug{}/Aug_{} clip style feature loss'.format(len(augnets), len(augnets)),
                #                   clip_style_feature_loss.item(), accumulated_iter)

                aug_fea1 = []
                images_fea = []

                for index,img in enumerate(aug_img):
                    img = img - text_featrues[index // 9]
                    img2 = images[index] - text_featrues[index // 9]
                    aug_fea1.append(img.view(1,-1))
                    images_fea.append(img2.view(1,-1))

                aug_fea = model.shared_network(torch.cat(aug_fea1).to(args.device))
                input_fea = model.shared_network(torch.cat(images_fea).to(args.device))

                feature_mse = augnet_criterion(aug_fea,input_fea)
                feature_loss = args.w_dist_input * torch.clamp(feature_mse, 0.2,10)
                loss_A += feature_loss - 1
                tb_log.add_scalar('train_Aug{}/Aug_{} cls feature loss'.format(len(augnets),len(augnets)), feature_loss.item(), accumulated_iter)


                if len(augnets) >= 1:
                    # distant from previous augnets
                    idx = np.random.randint(0, len(augnets))
                    aug_img_pre = augnets[idx](images.detach())

                    aug_fea2 = []

                    for index, img in enumerate(aug_img_pre):
                        img = img - text_featrues[index // 9]
                        aug_fea2.append(img.view(1, -1))

                    aug_fea_pre = model.shared_network(torch.cat(aug_fea2)).to(args.device)
                    loss_distant_preaug = torch.mean(F.cosine_similarity(aug_fea, aug_fea_pre, dim=1))  # smaller similarity
                    # losses.update({'loss_distant_preaug': loss_distant_preaug.item()})
                    distance_loss = args.w_dist_pre * torch.clamp(loss_distant_preaug, 0,1)
                    loss_A -= distance_loss - 1
                    tb_log.add_scalar('train_Aug{}/Aug_{} distance loss'.format(len(augnets),len(augnets)), distance_loss.item(),
                                      accumulated_iter)
                loss_A.backward()
                optimizerA.step()
                schedulerA.step()

                # -------- train augnet end --------

                # -------- step2 :train classifier start --------
                set_requires_grad([augnet], False)
                set_requires_grad([model], True)
                optimizer.zero_grad()

                pred = model(images1,images2)

                loss = criterion(pred,labels)
                tb_log.add_scalar('train/seen pred loss', loss.item(), accumulated_iter)

                # get augnet data
                n = 0
                aug_img = augnet(images)
                image_list = []
                image_combine = []
                new_image1 = []
                new_image2 = []
                new_label = []

                for index,img in enumerate(aug_img):
                    img = img - text_featrues[index // 9]
                    img2 = images[index] - text_featrues[index // 9]
                    image_list.append(img.view(1,-1))
                    image_list.insert(0,img2.view(1,-1))
                    n += 1
                    if n == 9:
                        indices = np.arange(len(pairs))
                        np.random.shuffle(indices)
                        shuffled_data = pairs[indices]
                        label_aug = pairs_labels[indices]

                        for x, y in shuffled_data:
                            new_image1.append(image_list[x])
                            new_image2.append(image_list[y])

                        new_label.append(torch.from_numpy(label_aug).type(torch.float32))
                        image_combine = image_combine + [image_list]
                        image_list = []

                        n = 0

                new_image1 = torch.cat(new_image1).to(args.device)
                new_image2 = torch.cat(new_image2).to(args.device)
                new_label = torch.cat(new_label).view(-1,1).to(args.device)
                # image_combine = torch.cat(image_combine).to(args.device)

                pred2 = model(new_image1, new_image2)
                aug_pred_loss = criterion(pred2, new_label)
                loss += aug_pred_loss
                tb_log.add_scalar('Aug Pred Loss/aug{} pred loss'.format(len(augnets)), aug_pred_loss.item(), accumulated_iter)

                # classification on previous aug data
                if epoch >= 1:
                    idx = np.random.randint(0, len(augnets))
                    aug_img_pre = augnets[idx](images)
                    n = 0
                    image_list = []
                    new_image11 = []
                    new_image22 = []
                    new_label22 = []

                    for index, img in enumerate(aug_img_pre):
                        img = img - text_featrues[index // 9]
                        # img2 = images[index] - text_featrues[index // 9]
                        image_list.append(img.view(1, -1))
                        # image_list.insert(0,img2.view(1, -1))

                        n += 1
                        if n == 9:

                            image_list = image_combine[(index+1)%9] + image_list

                            indices = np.arange(len(pairs2))
                            np.random.shuffle(indices)
                            shuffled_data = pairs2[indices]
                            label_aug = pairs_labels2[indices]

                            for x, y in shuffled_data:
                                new_image11.append(image_list[x])
                                new_image22.append(image_list[y])
                            n = 0

                            new_label22.append(torch.from_numpy(label_aug).type(torch.float32))
                            image_list = []

                    new_image11 = torch.cat(new_image11).to(args.device)
                    new_image22 = torch.cat(new_image22).to(args.device)
                    new_label22 = torch.cat(new_label22).view(-1,1).to(args.device)

                    pred3 = model(new_image11, new_image22)
                    aug_diff_pred_loss = criterion(pred3, new_label22)
                    loss += aug_diff_pred_loss
                    tb_log.add_scalar('Pred Aug Diff Loss/aug{} diff pred loss'.format(len(augnets)), aug_diff_pred_loss.item(),
                                      accumulated_iter)

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(criterion.parameters()), max_norm=1.0)
                optimizer.step()

                # measure total time
                torch.cuda.synchronize()
                batch_time.update(time.perf_counter() - tic)
                tic = time.perf_counter()

                accumulated_iter += 1

                # display
                if i % args.disp_iter == 0:
                    print('Epoch: [{}][{}/{}], batch time: {:.3f}, Data time: {:.3f},Aug loss:{:.4f}, loss: {:.4f}'
                          .format(epoch, j*len(train_loader)+i, num_aug_epoch*len(train_loader),
                                  batch_time.average(), dataload_time.average(),loss_A.item(), loss.item()))

                # add tensorboard
                # tb_log.add_scalar('train/Aug loss', loss_A.item(), accumulated_iter)
                # tb_log.add_scalar('train/loss', loss.item(), accumulated_iter)
                tb_log.add_scalar('train/epoch', epoch, accumulated_iter)

                pbar.update(1)

    return accumulated_iter,augnet

def plot_TSNE(X,Y,number):
    # 使用t-SNE进行降维 (将数据降到2维)
    tsne = TSNE(n_components=2, random_state=4396)
    X_tsne = tsne.fit_transform(X)

    # 创建散点图
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=[color_map[class_names[i]] for i in Y], s=25)

    # 手动为每个类定义标签
    handles = [plt.Line2D([], [], marker='o', color=fixed_palette[i], linestyle='', markersize=10) for i in
               range(len(class_names))]
    plt.legend(handles, class_names, title="Classes", loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6)
    # 去掉坐标轴的刻度，但保留边框
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    # 调整布局，增加底部空间，以显示完整的图例
    plt.subplots_adjust(bottom=0.2)
    plt.title('Aug Net {}'.format(number))
    # 显示图像
    plt.show()

def evaluate(model, val_loader, criterion, epoch, history, tb_log, count_iter, args, augnets):
    logging.info(f'Evaluating at {epoch} epochs...')

    loss_metrics = AverageMeter()

    history['feature_distance'][epoch] = []

    model.eval()

    X = []
    Y = []

    with torch.no_grad():
        with tqdm(total=len(val_loader), desc='validation', unit='batch', leave=False) as pbar:

            for i, batch_data in enumerate(val_loader):
                images1, images2, _, labels, _ = batch_data
                images1 = images1.to(args.device)
                images2 = images2.to(args.device)
                labels = labels.to(args.device)

                pred = model(images1,images2)

                eval_loss = criterion(pred,labels)

                loss_metrics.update(eval_loss.item())

                pbar.update(1)

    for item in data_list[:200]:
        img = np.load(os.path.join(data_dir, class_names[0], item))
        img = torch.from_numpy(img).type(torch.float32).to(args.device)

        img2 = np.load(os.path.join(data_dir, class_names[1], item))
        img2 = torch.from_numpy(img2).type(torch.float32).to(args.device)

        img3 = np.load(os.path.join(data_dir, class_names[2], item))
        img3 = torch.from_numpy(img3).type(torch.float32).to(args.device)

        with torch.no_grad():
            feature = model.shared_network(img)
            X.append(feature[0].cpu().detach().numpy())
            Y.append(0)

            feature = model.shared_network(img2)
            X.append(feature[0].cpu().detach().numpy())
            Y.append(1)

            feature = model.shared_network(img3)
            X.append(feature[0].cpu().detach().numpy())
            Y.append(2)

            for augnet in augnets:
                aug_feature = augnet(img)
                aug_feature = model.shared_network(aug_feature)
                X.append(aug_feature[0].cpu().detach().numpy())
                Y.append(3)

                aug_feature2 = augnet(img2)
                aug_feature2 = model.shared_network(aug_feature2)
                X.append(aug_feature2[0].cpu().detach().numpy())
                Y.append(4)

                aug_feature3 = augnet(img3)
                aug_feature3 = model.shared_network(aug_feature3)
                X.append(aug_feature3[0].cpu().detach().numpy())
                Y.append(5)

    X = np.array(X)
    Y = np.array(Y)


    # without distribute
    history['eval_loss'].append(loss_metrics.average())
    tb_log.add_scalar('eval/loss', eval_loss, count_iter)
    print('Evaluation Summary: Epoch: {}, Loss: {:.4f}'.format(epoch, loss_metrics.average()))

    print('Draw TSNE visual ')
    plot_TSNE(X, Y, len(augnets))


def main(args):
    tb_log = SummaryWriter(log_dir=args.tensorboard_dir)

    model = SiameseNetwork(args.feature_dim)

    model.to(args.device)

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = ContrastDataset(args.data_json, transform=data_transform)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.1 * dataset_size))

    train_indices, val_indices = indices[split:], indices[:split]

    # 创建 samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # 创建 DataLoader
    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn)
    val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_sampler, collate_fn=collate_fn)

    criterion = nn.BCELoss()
    augnet_criterion = torch.nn.MSELoss()

    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=0.7)

    logging.info(f'''Starting training:
            Epochs:          {args.epochs}
            Batch size:      {args.batch_size}
            Learning rate:   {args.lr}
        ''')

    start_epoch = count_iter = 0

    # initialize checkpoint package
    history = {'eval_loss': [],'feature_distance':{}}

    augnets = []

    for epoch in range(start_epoch,args.epochs):
        model.train()
        start = time.time()

        logging.info('-' * 70)
        logging.info('Training...')

        # train one epoch
        count_iter, augnet = train_one_epoch(model, train_loader, optimizer, criterion, epoch, count_iter, tb_log, args, augnets, augnet_criterion)
        augnets.append(augnet)


        logging.info(f'Train Summary | End of Epoch {epoch} | Time {time.time() - start:.2f}s')

        scheduler.step()

        # # Evaluation round
        logging.info('-' * 70)
        logging.info('Evaluating...')
        evaluate(model, val_loader, criterion, epoch, history, tb_log, count_iter, args, augnets)

        if args.local_rank == 0:
            # checkpointing
            checkpoint(model, criterion, history, epoch, optimizer, count_iter, args)
            augnet_name = 'augnet_' + str(epoch) + '.pth'
            torch.save(augnet.state_dict(), '{}/package_{}'.format(args.save_ckpt, augnet_name))

if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Train a Style_CLIP")

    # 添加参数
    parser.add_argument('--model', type=str, default='hr_net', help="")
    parser.add_argument('--disp_iter', type=int, default=50, help="")
    parser.add_argument('--local_rank', type=int, default=0, help="")
    parser.add_argument('--feature_dim', type=int, default=1024, help="")
    parser.add_argument('--batch_size', type=int, default=20, help="batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--augnet_lr', type=float, default=1e-3, help="augnet learning rate")
    parser.add_argument('--lr_step_size', type=int, default=4, help="learning rate decay")
    parser.add_argument('--augnet_step_size', type=int, default=2000, help="augnet learning rate decay")
    parser.add_argument('--w_dist_pre', type=float, default=1, help="augnet learning param")
    parser.add_argument('--w_dist_input', type=float, default=1e-1, help="augnet learning param")
    parser.add_argument('--w_dist_inside_aug_fea', type=float, default=1e-3, help="augnet learning param")
    parser.add_argument('--w_dist_clip_feature', type=float, default=10, help="augnet learning param")

    parser.add_argument('--epochs', type=int, default=40, help="")
    parser.add_argument('--tensorboard_dir', type=str,default='./logs/test_sdv21_lcm_sdv14_20batch_40epoch_augclsdis10' ,help="")
    parser.add_argument('--data_json', type=str, default='/home/jxq/Datasets/clip_encoder_feature', help="")
    parser.add_argument('--device', type=str, default='cuda:0', help="")
    parser.add_argument('--save_ckpt', type=str, default='./ckpt/test_sdv21_lcm_sdv14_20batch_40epoch_augclsdis10', help="")
    parser.add_argument('--best_loss', dest='best_loss',help='best loss for evaluation', default=float("inf"))



    # 解析命令行参数
    args = parser.parse_args()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main(args)