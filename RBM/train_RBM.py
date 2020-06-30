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
    print(len(mnist_training), len(mnist_testing))
    print(type(mnist_training[0]), len(mnist_training[0]))
    print(type(mnist_training[0][0]), type(mnist_training[0][1]))
    print(mnist_training[0][0].shape)
    print(mnist_training[0][0].numpy()[0].shape)

    plt.imshow(mnist_training[0][0].numpy()[0])
    plt.colorbar()
    plt.show()