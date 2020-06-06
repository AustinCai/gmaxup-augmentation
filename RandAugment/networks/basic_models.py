from torch import nn 
import torch.nn.functional as F

# helpers for train_model.py and train_gmaxup.py =======================================================

class Linear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, xb):
        return self.lin(xb)


class SmallNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.l1 = nn.Linear(in_channels, 32)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, out_channels) 

    def forward(self, xb):
        a1 = F.relu(self.l1(xb))
        a2 = F.relu(self.l2(a1))
        return self.l3(a2)


class LargeNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.l1 = nn.Linear(in_channels, 1024)
        self.l2 = nn.Linear(1024, 1024)
        self.l3 = nn.Linear(1024, out_channels) 

        # self.do = nn.Dropout(0.3)
        # self.bn = nn.BatchNorm2d(1024)

    def forward(self, xb):
        a1 = F.relu(self.l1(xb))
        # a1 = self.bn(a1)
        a2 = F.relu(self.l2(a1))
        # a2 = self.bn(a2)
        # a2 = self.do(a2)
        return self.l3(a2)


class SmallCNN(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 18, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(18, 18, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(int(18 * (32/4)**2), 64)
        self.fc2 = nn.Linear(64, out_channels)

    def forward(self, x):
        a1 = F.relu(self.conv1(x))
        a1 = self.pool(a1)
        a2 = F.relu(self.conv2(a1))
        a2 = self.pool(a2)
        a3 = a2.view(-1, int(self.out_pool / self.stride_pool**2))
        a3 = F.relu(self.fc1(a3))
        return self.fc2(a3)


# https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition
# -task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/
class BestCNN(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.do1 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.do2 = nn.Dropout(0.3)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.do3 = nn.Dropout(0.4)

        self.fc1 = nn.Linear(8192, out_channels)

    def forward(self, x):
        a1 = F.relu(self.conv1(x))
        a1 = self.bn1(a1)
        a2 = F.relu(self.conv2(a1))
        a2 = self.bn1(a2)
        a3 = self.pool(a2)
        a3 = self.do1(a3)

        a4 = F.relu(self.conv3(a3))
        a4 - self.bn2(a4)
        a5 = F.relu(self.conv4(a4))
        a5 = self.bn2(a5)
        a6 = self.pool(a5)
        a6 = self.do2(a6)

        a7 = F.relu(self.conv5(a6))
        a7 = self.bn3(a7)
        a8 = F.relu(self.conv6(a7))
        a8 = self.bn3(a8)
        a8 = self.do3(a8)

        a9 = a8.view(-1, 8192)
        return self.fc1(a9)

# helpers for train_model.py ============================================================================
# =======================================================================================================


# helpers for train_gmaxup.py ===========================================================================
# =======================================================================================================


# helpers for testing ===================================================================================
# =======================================================================================================

def test():
    pass


if __name__ == "__main__":
    test()