import torch
import torch.nn as nn
import torchvision.models as models


# --------- LeNet-5 for MNIST --------- #
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(torch.relu(self.conv2(x)), 2)
        x = torch.max_pool2d(torch.relu(self.conv3(x)), 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# --------- ResNet-18 for CIFAR-10 --------- #
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=False, num_classes=10)

    def forward(self, x):
        return self.model(x)


# --------- VGG-16 for CIFAR-100 --------- #
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.model = models.vgg16(pretrained=False, num_classes=100)

    def forward(self, x):
        return self.model(x)
