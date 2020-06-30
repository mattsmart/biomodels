import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

DIR_DATA = 'data'
CPU_THREADS = 8
BATCH_SIZE = 4


if __name__ == '__main__':
    # data structure: list of ~60,000 2-tuples: "image" and integer label
    mnist_training = torchvision.datasets.MNIST(root=DIR_DATA, train=True, transform=torchvision.transforms.ToTensor(), download=True)
    mnist_testing = torchvision.datasets.MNIST(root=DIR_DATA, train=False, transform=torchvision.transforms.ToTensor(), download=True)
    data_loader = torch.utils.data.DataLoader(mnist_training, batch_size=BATCH_SIZE, shuffle=True, num_workers=CPU_THREADS)

    print('TODO')
