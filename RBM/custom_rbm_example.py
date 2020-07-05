import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
import torchvision.datasets
import torchvision.models
import torchvision.transforms

from custom_rbm import RBM_custom, RBM_gaussian_custom
from data_process import image_data_collapse, binarize_image_data
from RBM_train import build_rbm_hopfield
from RBM_assess import plot_confusion_matrix, rbm_features_MNIST
from settings import MNIST_BINARIZATION_CUTOFF


"""
WHAT'S CHANGED:
- removed cuda lines
- removed momentum
- updates actually sample hidden and visible values (instead of passing bernoulli probabilities) 
- flip state to +1, -1
- TODO: remove or change regularization: none, L1 (and scale), L2 (and scale)
- (?) remove weight decay
- (?) remove momentum
- (?) remove applied fields?
- (?) augment learning rate
- (?) augment logistic regression
- (?) option for binary or gaussian hidden nodes
"""


########## CONFIGURATION ##########
BATCH_SIZE = 64
VISIBLE_UNITS = 784  # 28 x 28 images
HIDDEN_UNITS = 10  # was 128 but try 10
CD_K = 2
EPOCHS = 0  # was 10
DATA_FOLDER = 'data'
GAUSSIAN_RBM = True
LOAD_INIT_WEIGHTS = True

if RBM_gaussian_custom:
    RBM = RBM_gaussian_custom
else:
    RBM = RBM_custom

if LOAD_INIT_WEIGHTS:
    assert HIDDEN_UNITS == 10

########## LOADING DATASET ##########
print('Loading dataset...')
train_dataset = torchvision.datasets.MNIST(root=DATA_FOLDER, train=True, transform=torchvision.transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_dataset = torchvision.datasets.MNIST(root=DATA_FOLDER, train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

########## TRAINING RBM ##########
print('Training RBM...')
rbm = RBM(VISIBLE_UNITS, HIDDEN_UNITS, CD_K, load_init_weights=LOAD_INIT_WEIGHTS)
for epoch in range(EPOCHS):
    epoch_error = 0.0
    for batch, _ in train_loader:
        batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data
        batch = (batch > MNIST_BINARIZATION_CUTOFF).float()    # convert to 0,1 form
        batch = -1 + batch * 2                                 # convert to -1,1 form
        batch_error = rbm.contrastive_divergence(batch)
        epoch_error += batch_error
    print('Epoch Error (epoch=%d): %.4f' % (epoch, epoch_error))

########## EXTRACT FEATURES ##########
print('Extracting features...')
train_features = np.zeros((len(train_dataset), HIDDEN_UNITS))
train_labels = np.zeros(len(train_dataset))
test_features = np.zeros((len(test_dataset), HIDDEN_UNITS))
test_labels = np.zeros(len(test_dataset))
for i, (batch, labels) in enumerate(train_loader):
    batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data
    batch = (batch > MNIST_BINARIZATION_CUTOFF).float()   # convert to 0,1 form
    batch = 2 * batch - 1                                 # convert to -1,1 form
    train_features[i * BATCH_SIZE:i * BATCH_SIZE + len(batch), :] = rbm.sample_hidden(batch)
    train_labels[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = labels.numpy()

for i, (batch, labels) in enumerate(test_loader):
    batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data
    batch = (batch > MNIST_BINARIZATION_CUTOFF).float()   # convert to 0,1 form
    batch = 2 * batch - 1                                 # convert to -1,1 form
    test_features[i*BATCH_SIZE:i*BATCH_SIZE+len(batch), :] = rbm.sample_hidden(batch)
    test_labels[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = labels.numpy()

########## CLASSIFICATION ##########
clf = LogisticRegression(C=1e5, multi_class='multinomial', penalty='l1', solver='saga', tol=0.1)
#clf = LogisticRegression(solver='newton-cg', tol=1)
#clf = LogisticRegression()
print('Classifying...')
clf.fit(train_features, train_labels)
predictions = clf.predict(test_features).astype(int)

########## CONFUSION MATRIX ##########
confusion_matrix = np.zeros((10, 10), dtype=int)
matches = [False for _ in test_dataset]
for idx, pair in enumerate(test_dataset):
    if pair[1] == predictions[idx]:
        matches[idx] = True
    confusion_matrix[pair[1], predictions[idx]] += 1
print("Successful test cases: %d/%d (%.3f)" % (matches.count(True), len(matches), float(matches.count(True) / len(matches))))
plot_confusion_matrix(confusion_matrix)
