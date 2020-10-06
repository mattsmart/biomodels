import matplotlib.pyplot as plt
import numpy as np

from data_process import data_mnist, binarize_image_data, image_data_collapse
from settings import MNIST_BINARIZATION_CUTOFF


def PCA_on_dataset(num_samples, num_features, dataset=None, binarize=True):
    def get_X():
        X = np.zeros((len(dataset), num_features))
        for idx, pair in enumerate(dataset):
            elem_arr, elem_label = pair
            preprocessed_input = image_data_collapse(elem_arr)
            if binarize:
                preprocessed_input = binarize_image_data(preprocessed_input, threshold=MNIST_BINARIZATION_CUTOFF)
            features = preprocessed_input
            X[idx, :] = features
        return X

    def covariance_scaled(X):
        # see Eq. (1) of https://arxiv.org/pdf/1104.3665.pdf
        # NOTE: since X can have features which are constant, the diagonals of gamma can be 0 instead of 1
        num_samples, num_features = X.shape
        means = np.mean(X, axis=0)
        assert binarize
        sqrt_one_minus_meansqr = np.sqrt(1 - means ** 2)  # assert binarize bc stdev bernoulli
        C = np.dot(X.T, X) / num_samples
        Gamma = np.zeros((num_features, num_features))
        for i in range(num_features):
            for j in range(i + 1):
                num = C[i, j] - means[i] * means[j]
                den = sqrt_one_minus_meansqr[i] * sqrt_one_minus_meansqr[j]
                if den == 0:
                    val = 0
                else:
                    val = num / den

                if i == j:
                    Gamma[i, j] = val
                else:
                    Gamma[i, j] = val
                    Gamma[j, i] = val
        return Gamma

    def build_X_random():
        # Note: raw pixel fata is bimodal concentrates near 0, 1
        # TODO consider other random distributions
        #  e.g. bernoulli with d% where (scalar or arr) d = mean of data pixels on/off
        assert binarize
        # https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.binomial.html
        param = 0.5  # TODO currently every pixel is coin flip
        X = np.random.binomial(1, param, (num_samples, num_features))
        X = 2 * X - 1
        return X

    def eval_evec(arr):
        eig_vals, eig_vecs = np.linalg.eig(arr)
        sort_perm = eig_vals.argsort()[::-1]
        evals_sorted = np.sort(eig_vals)[::-1]
        evecs_sorted = eig_vecs[:, sort_perm]
        return evals_sorted, evecs_sorted

    print("Prepare dataset for PCA")
    if dataset is not None:
        assert len(dataset) == num_samples
        X = get_X()
    else:
        X = build_X_random().astype("float64")  # observed int32 mult MUCH slower than floats on desktop...
    print("Build correlation matrix")
    Gamma = covariance_scaled(X)
    print("Diagonalize")
    evals, evecs = eval_evec(Gamma)

    return X, Gamma, evals, evecs


def PCA_evals_info(evals):
    evals_ratio = evals / np.sum(evals)
    evals_cumsum = np.cumsum(evals_ratio)

    plt.hist(evals, bins=50)
    plt.show()

    plt.plot(evals_ratio)
    plt.yscale("log")
    plt.show()

    plt.plot(evals_cumsum)
    plt.show()
    return evals_ratio, evals_cumsum


def get_patterns_from_PCA(means, evals, evecs, p=10, pHat=10):
    # TODO remove zero evals, possibly when/before generating gamma (remove pixels that are constant)
    # TODO use to reconstruct J based on the reference, since the xi are not binary
    num_features = len(means)
    assert p + pHat < num_features
    xi_arr = np.zeros((num_features, p))
    xiHat_arr = np.zeros((num_features, pHat))

    evals_attract = evals[0:p]
    evecs_attract = evecs[:, 0:p]

    evals_repulse = evals[-pHat:]  # TODO check -- need pHat smallest nonzero evals
    evecs_repulse = evecs[:, -pHat:]  # TODO check -- need pHat smallest nonzero evals
    #evals_repulse = evals[-73:-70]  # TODO check
    #evecs_repulse = evecs[:, -73:-70]  # TODO check

    for i in range(num_features):
        # den = np.sqrt(1 - means[i]**2)
        den = np.sqrt(1 - means[i] ** 2)
        for mu in range(p):
            prefactor = np.sqrt(num_features * (1 - 1 / evals_attract[mu]))
            xi_arr[i, mu] = prefactor / den * evecs_attract[i, mu]
        for mu in range(pHat):
            prefactor = np.sqrt(num_features * (1 / evals_repulse[mu] - 1))
            xiHat_arr[i, mu] = prefactor / den * evecs_repulse[i, mu]

    return xi_arr, xiHat_arr


if __name__ == '__main__':
    TRAINING, TESTING = data_mnist()
    dataset = TRAINING[0:10000]
    X, Gamma, evals, evecs = PCA_on_dataset(len(dataset), 28 ** 2, dataset=dataset, binarize=True)
    X_rand, Gamma_rand, evals_rand, evecs_rand = PCA_on_dataset(len(dataset), 28 ** 2, dataset=None, binarize=True)

    evals_ratio, evals_cumsum = PCA_evals_info(evals)

    plt.plot(evals_ratio[0:200])
    plt.yscale("log")
    plt.show()

    evals_rand_ratio, evals_rand_cumsum = PCA_evals_info(evals_rand)

    plt.plot(evals_ratio[0:200])
    plt.yscale("log")
    plt.show()

    means = np.mean(X, axis=0)
    xi_arr, _ = get_patterns_from_PCA(means, evals, evecs, p=10, pHat=0)
    plt.imshow(xi_arr[:,1].reshape(28,28))
    plt.colorbar()
    plt.show()