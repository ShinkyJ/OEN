import torch.nn as nn
import torch

#增广网络
class augment_net(nn.Module):
    def __init__(self, input_dim):
        super(augment_net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        return x

# 共享网络
class SharedNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SharedNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        return x

# Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self, input_dim, hrnet_pretrain = '/home/jxq/projects/torch113/HRNet-Image-Classification/ckpt/hrnet_w18_small_model_v1.pth'):
        super(SiameseNetwork, self).__init__()

        self.shared_network = SharedNetwork(input_dim)
        self.fc_out = nn.Linear(2048, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        # x1 = self.net(x1)
        # x2 = self.net(x2)
        out1 = self.shared_network(x1)
        out2 = self.shared_network(x2)
        distance = torch.abs(out1 - out2)
        output = self.fc_out(distance)
        output = self.sigmoid(output)
        return output

if __name__ == '__main__':
    model = SiameseNetwork(1024)
    data = torch.randn(4, 1024)
    data1 = torch.randn(4, 1024)
    pred = model(data,data1)
    print(1)