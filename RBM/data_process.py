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


def data_dict_mnist_inspect(data_dict, category_counts):
    data_dimension = (28,28)
    data_dimension_collapsed = data_dimension[0] * data_dimension[1]

    """
    print("CORRELATIONS")
    tot = len(list(data_dict.keys()))
    all_data_correlation = np.zeros((tot, tot))
    start_idx = [0]*10
    for idx in range(1,10):
        start_idx[idx] = start_idx[idx-1] + category_counts[idx-1]
    for i in range(10):
        data_i = data_dict[i].reshape(784, category_counts[i])
        i_a = start_idx[i]
        if i == 9:
            i_b = len(data)
        else:
            i_b = start_idx[i + 1]
        for j in range(i+1):
            print(i,j)
            data_j = data_dict[j].reshape(784, category_counts[j])
            corr = np.dot(data_i.T, data_j)
            j_a = start_idx[j]
            if j == 9:
                j_b = len(data)
            else:
                j_b = start_idx[j + 1]
            all_data_correlation[i_a:i_b, j_a:j_b] = corr
            all_data_correlation[j_a:j_b, i_a:i_b] = corr.T
            plt.imshow(corr)
            plt.colorbar()
            plt.title('%d,%d (corr)' % (i,j))
            plt.show()
    plt.imshow(all_data_correlation)  # out of memory error, show only in parts above
    plt.colorbar()
    plt.title('all data (corr)')
    plt.show()
    """

    for idx in range(10):
        print(idx)
        category_amount = category_counts[idx]
        data_idx_collapsed = data_dict[idx].reshape(data_dimension_collapsed, category_amount)

        # assumes data will be binarized
        data_idx_collapsed[data_idx_collapsed > MNIST_BINARIZATION_CUTOFF] = 1
        data_idx_collapsed[data_idx_collapsed < MNIST_BINARIZATION_CUTOFF] = 0

        print("DISTANCES")
        category_amount_redux = int(category_amount)  # TODO enlarge
        distance_arr = np.zeros((category_amount_redux, category_amount_redux))
        for i in range(category_amount_redux):
            a = data_idx_collapsed[:, i]
            for j in range(i + 1):
                b = data_idx_collapsed[:, j]
                val = np.count_nonzero(a != b)
                distance_arr[i, j] = val
                distance_arr[j, i] = val

        print("MANUAL DENDROGRAM")
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform
        condensed_dist = squareform(distance_arr)

        Z = linkage(condensed_dist, 'ward')
        fig = plt.figure(figsize=(25, 10))
        dn = dendrogram(Z)
        plt.title(idx)
        plt.savefig(DIR_OUTPUT + os.sep + 'dend_%d.png' % idx)
    return


def data_dict_mnist_detailed(data_dict, category_counts):
    """
    Idea here is to further divide the patterns into subtypes to aid classification: (7_0, 7_1, etc)
        - should the binarization already be happening or not yet?
    Intermediate form of data_dict_detailed:
        {6: {0: array, 1: array}}  (representing 6_0, 6_1 subtypes for example)
    Returned form:
        {'6_0': array, '6_1': array}  (representing 6_0, 6_1 subtypes for example)
    """
    data_dimension = (28,28)
    data_dimension_collapsed = data_dimension[0] * data_dimension[1]
    data_dict_detailed = {idx: {} for idx in range(10)}

    for idx in range(10):
        print(idx)
        category_amount = category_counts[idx]
        data_idx_collapsed = data_dict[idx].reshape(data_dimension_collapsed, category_amount)

        # assumes data will be biinarized
        data_idx_collapsed[data_idx_collapsed > MNIST_BINARIZATION_CUTOFF] = 1
        data_idx_collapsed[data_idx_collapsed < MNIST_BINARIZATION_CUTOFF] = 0

        print("Auto AgglomerativeClustering")
        # note euclidean is sqrt of 01 flip distance (for binary data), ward seems best, threshold 24 gave 2-5 clusters per digit
        print(data_idx_collapsed.shape)
        from sklearn.cluster import AgglomerativeClustering
        cluster = AgglomerativeClustering(n_clusters=K_PATTERN_DIV, affinity='euclidean', linkage='ward', distance_threshold=None)
        #cluster = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='ward', distance_threshold=24)
        cluster_labels = cluster.fit_predict(data_idx_collapsed.T)

        # pre-allocate subcategory arrays and fill in
        unique, counts = np.unique(cluster_labels, return_counts=True)
        sublabeldict = dict(zip(unique, counts))
        print(sublabeldict)
        for k in sublabeldict.keys():
            sublabel_indices = np.argwhere(cluster_labels == k)
            data_dict_detailed[idx][k] = np.squeeze( data_idx_collapsed[:, sublabel_indices] ).reshape(28, 28, len(sublabel_indices))

    data_dict_detailed_flat = {}
    category_counts_detailed_flat = {}
    for idx in range(10):
        for subkey in data_dict_detailed[idx].keys():
            newkey = '%d_%d' % (idx, subkey)
            data_dict_detailed_flat[newkey] = data_dict_detailed[idx][subkey]
            category_counts_detailed_flat[newkey] = data_dict_detailed[idx][subkey].shape[2]

    return data_dict_detailed_flat, category_counts_detailed_flat


def hopfield_mnist_patterns(data_dict, category_counts, pattern_threshold=PATTERN_THRESHOLD):
    """
    data: list of tuples (numpy array, labe;)
    pattern_threshold: threshold for setting value to 1 in the category-specific voting for pixel values
    Returns:
        xi: N x P binary pattern matrix
    """
    keys = sorted(data_dict.keys())
    pattern_idx_to_labels = {idx: keys[idx] for idx in range(len(keys))}

    data_dimension = data_dict[keys[0]].shape[:2]

    # testing additional pre-binarization step
    for key in keys:
        data_dict[key] = binarize_image_data(data_dict[key], threshold=MNIST_BINARIZATION_CUTOFF)

    print("Forming %d MNIST patterns" % len(keys))
    xi_images = np.zeros((*data_dimension, len(keys)), dtype=int)
    for idx, key in enumerate(keys):
        samples = data_dict[key]
        samples_avg = np.sum(samples, axis=2) / category_counts[key]
        samples_avg[samples_avg <= pattern_threshold] = -1  # samples_avg[samples_avg <= pattern_threshold] = -1
        samples_avg[samples_avg > pattern_threshold] = 1
        xi_images[:, :, idx] = samples_avg
    xi_collapsed = image_data_collapse(xi_images)
    print("xi_collapsed.shape", xi_collapsed.shape)
    return xi_images, xi_collapsed, pattern_idx_to_labels


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
