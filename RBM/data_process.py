import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

DIR_DATA = 'data'
CPU_THREADS = 8
BATCH_SIZE = 4

SYNTHETIC_DIM = 8
SYNTHETIC_SAMPLES = 10000
SYNTHETIC_NOISE_VALID = ['symmetric']
SYNTHETIC_SAMPLING_VALID = ['balanced']
SYNTHETIC_DATASPLIT = ['balanced']

"""
noise 'symmetric': the noise for each pattern basin is symmetric
   - Q1: does it matter then that each pattern is uncorrelated? should sample noise differently if correlated?  
sampling 'balanced': 50% of samples drawn from each pattern basin (for 2 patterns) 
"""


def data_mnist():
    # data structure: list of ~60,000 2-tuples: "image" and integer label
    training = torchvision.datasets.MNIST(root=DIR_DATA, train=True, transform=torchvision.transforms.ToTensor(), download=True)
    testing = torchvision.datasets.MNIST(root=DIR_DATA, train=False, transform=torchvision.transforms.ToTensor(), download=True)
    #data_loader = torch.utils.data.DataLoader(training, batch_size=BATCH_SIZE, shuffle=True, num_workers=CPU_THREADS)  # TODO read
    return training, testing


def data_synthetic_dual(num_samples=SYNTHETIC_SAMPLES, noise='symmetric', sampling='balanced', datasplit='balanced'):
    # 1. specify patterns
    assert SYNTHETIC_DIM == 8  # TODO alternative even integers? maybe very large is better (like images)?
    pattern_A = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=int)  # label 0
    pattern_B = np.array([1, 1, 1, 1,  1,  1,  1,  1], dtype=int)  # label 1

    # 2. create data distribution with labels -- TODO care
    #   step 1: have some true distribution in mind (e.g. two equal - or not - minima at the patterns; noise matters)
    #   step 2: generate M samples which follow from that distribution
    data_array = np.zeros((SYNTHETIC_DIM, SYNTHETIC_SAMPLES), dtype=int)
    data_labels = np.zeros(SYNTHETIC_SAMPLES, dtype=int)
    if noise == 'symmetric':
        assert sampling == 'balanced'
        assert num_samples % 2 == 0
        labels_per_pattern = int(num_samples/2)
        for idx in range(labels_per_pattern):
            pattern_A_sample = None  # TODO
            data_array[:, idx] = pattern_A_sample
            data_labels[idx] = 0

            pattern_B_sample_idx = labels_per_pattern
            pattern_B_sample = None  # TODO
            data_array[:, pattern_B_sample_idx] = pattern_B_sample
            data_labels[pattern_B_sample_idx] = 1
    else: assert noise in SYNTHETIC_NOISE_VALID

    # 3. save the data
    # TODO save to DIR_DATA subfolder (diff for each noise/size case?)

    # 4. split into training and testing (symmetrically or not?)
    if datasplit == 'balanced':
        # TODO
        training = None
        testing = None
    else: assert datasplit in SYNTHETIC_DATASPLIT

    return training, testing


if __name__ == '__main__':
    mnist_training, mnist_testing = data_mnist()
    print(len(mnist_training), len(mnist_testing))
    print(type(mnist_training[0]), len(mnist_training[0]))
    print(type(mnist_training[0][0]), type(mnist_training[0][1]))
    print(mnist_training[0][0].shape)
    print(mnist_training[0][0].numpy()[0].shape)

    plt.imshow(mnist_training[0][0].numpy()[0])
    plt.colorbar()
    plt.show()
