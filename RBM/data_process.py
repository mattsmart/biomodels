import matplotlib.pyplot as plt
import numpy as np
import os
import torchvision
from settings import DIR_DATA, DIR_MODELS, DIR_OUTPUT, SYNTHETIC_DIM, SYNTHETIC_SAMPLES, SYNTHETIC_NOISE_VALID, \
    SYNTHETIC_SAMPLING_VALID, SYNTHETIC_DATASPLIT, MNIST_BINARIZATION_CUTOFF, PATTERN_THRESHOLD, K_PATTERN_DIV

"""
noise 'symmetric': the noise for each pattern basin is symmetric
   - Q1: does it matter then that each pattern is uncorrelated? should sample noise differently if correlated?  
sampling 'balanced': 50% of samples drawn from each pattern basin (for 2 patterns) 
"""


def binarize_image_data(numpy_obj, threshold=MNIST_BINARIZATION_CUTOFF):
    numpy_obj[numpy_obj <= threshold] = 0
    numpy_obj[numpy_obj > 0] = 1  # now 0, 1
    numpy_obj.astype(int)
    numpy_obj = 2 * numpy_obj - 1  # +1, -1 convention
    return numpy_obj


def torch_image_to_numpy(torch_tensor, binarize=False):
    numpy_obj = torch_tensor.numpy()[0]
    if binarize:
        numpy_obj = binarize_image_data(numpy_obj)
    return numpy_obj


def data_mnist(binarize=False):
    # data structure: list of ~60,000 2-tuples: "image" and integer label
    training = torchvision.datasets.MNIST(root=DIR_DATA, train=True, transform=torchvision.transforms.ToTensor(), download=True)
    testing = torchvision.datasets.MNIST(root=DIR_DATA, train=False, transform=torchvision.transforms.ToTensor(), download=True)
    #data_loader = torch.utils.data.DataLoader(training, batch_size=BATCH_SIZE, shuffle=True, num_workers=CPU_THREADS)  # TODO read

    print("Processing MNIST data: numpy_binarize =", binarize)
    training = [(torch_image_to_numpy(elem[0], binarize=binarize), elem[1]) for elem in training]
    testing = [(torch_image_to_numpy(elem[0], binarize=binarize), elem[1]) for elem in testing]

    return training, testing


def image_data_collapse(data):
    if len(data.shape) == 3:
        return data.reshape(-1, data.shape[-1])
    else:
        assert len(data.shape) == 2
        return data.flatten()


def samples_per_category(data):
    category_counts = {}
    for pair in data:
        if pair[1] in category_counts.keys():
            category_counts[pair[1]] += 1
        else:
            category_counts[pair[1]] = 1
    return category_counts


def data_dict_mnist(data):
    data_dimension = data[0][0].shape
    category_counts = samples_per_category(data)
    print("category_counts:\n", category_counts)
    print("Generating MNIST data dict")
    label_counter = {idx: 0 for idx in range(10)}
    data_dict = {idx: np.zeros((data_dimension[0], data_dimension[1], category_counts[idx])) for idx in range(10)}
    for pair in data:
        label = pair[1]
        category_idx = label_counter[label]
        label_counter[label] += 1
        category_array = data_dict[label]
        category_array[:, :, category_idx] = pair[0][:,:]
    return data_dict, category_counts


def hopfield_mnist_patterns(data_dict, category_counts, pattern_threshold=PATTERN_THRESHOLD):
    """
    data: list of tuples (numpy array, labe;)
    pattern_threshold: threshold for setting value to 1 in the category-specific voting for pixel values
    Returns:
        xi: N x P binary pattern matrix
    """
    data_dimension = data_dict[0].shape[:2]
    # testing additional pre-binarization step
    for i in range(10):
        for j in range(category_counts[i]):
            data_dict[i][:, :, j] = binarize_image_data(data_dict[i][:, :, j], threshold=MNIST_BINARIZATION_CUTOFF)
    print("Forming the 10 MNIST patterns")
    xi_images = np.zeros((*data_dimension, 10), dtype=int)
    for idx in range(10):
        samples = data_dict[idx]
        samples_avg = np.sum(samples, axis=2) / category_counts[idx]
        samples_avg[samples_avg <= pattern_threshold] = -1  # samples_avg[samples_avg <= pattern_threshold] = -1
        samples_avg[samples_avg > pattern_threshold] = 1
        xi_images[:, :, idx] = samples_avg
    xi_collapsed = image_data_collapse(xi_images)
    print("xi_collapsed.shape", xi_collapsed.shape)
    return xi_images, xi_collapsed


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
    # get data
    mnist_training, mnist_testing = data_mnist()

    inspect_data_dict = True
    if inspect_data_dict:
        data_dict, category_counts = data_dict_mnist(mnist_training)
        data_dict_detailed, category_counts_detailed = data_dict_mnist_detailed(data_dict, category_counts)
        xi_images, xi_collapsed, pattern_idx_to_labels = hopfield_mnist_patterns(data_dict, category_counts, pattern_threshold=0.0)
        for idx in range(xi_images.shape[-1]):
            plt.imshow(xi_images[:, :, idx])

    simple_pattern_vis = True
    if simple_pattern_vis:
        print("Plot hopfield patterns from 'voting'")
        data_dict, category_counts = data_dict_mnist(mnist_training)
        xi_mnist, _ = hopfield_mnist_patterns(data_dict, category_counts, pattern_threshold=0.0)
        for idx in range(10):
            plt.imshow(xi_mnist[:, :, idx])
            plt.colorbar()
            plt.show()
    else:
        thresholds = [-0.3, -0.2, -0.1, 0.0, 0.1]
        print("Grid of pattern subplots for varying threshold param", thresholds)
        fig, ax_arr = plt.subplots(len(thresholds), 10)
        for p, param in enumerate(thresholds):
            xi_mnist, _ = hopfield_mnist_patterns(data_dict, category_counts, pattern_threshold=param)
            xi_mnist = xi_mnist.astype(int)
            for idx in range(10):
                ax_arr[p, idx].imshow(xi_mnist[:, :, idx], interpolation='none')
                for i in range(28):
                    for j in range(28):
                        if xi_mnist[i, j, idx] not in [-1,1]:
                            print(xi_mnist[i, j, idx])
                ax_arr[p, idx].set_xticklabels([])
                ax_arr[p, idx].set_yticklabels([])
        plt.suptitle('Top to bottom thresholds: %s' % thresholds)
        plt.show()
