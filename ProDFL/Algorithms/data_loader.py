import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# --------- Data Loading for MNIST --------- #
def load_mnist_data(batch_size=64):
    """
    Loads the MNIST dataset and returns the train and test DataLoader.
    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Loading the MNIST training and test datasets
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Create DataLoader for MNIST training and test datasets
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader


# --------- Data Loading for CIFAR-10 --------- #
def load_cifar10_data(batch_size=64):
    """
    Loads the CIFAR-10 dataset and returns the train and test DataLoader.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # For color images (CIFAR-10)
    ])

    # Loading the CIFAR-10 training and test datasets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Create DataLoader for CIFAR-10 training and test datasets
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader


# --------- Data Loading for CIFAR-100 --------- #
def load_cifar100_data(batch_size=64):
    """
    Loads the CIFAR-100 dataset and returns the train and test DataLoader.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # For color images (CIFAR-100)
    ])

    # Loading the CIFAR-100 training and test datasets
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    # Create DataLoader for CIFAR-100 training and test datasets
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader
