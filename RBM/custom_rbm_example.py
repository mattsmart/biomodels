import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import torch
import torchvision.datasets
import torchvision.models
import torchvision.transforms

from AIS import esimate_logZ_with_AIS, get_obj_term_A
from custom_rbm import RBM_custom, RBM_gaussian_custom
from data_process import image_data_collapse, binarize_image_data, data_mnist
from RBM_train import build_rbm_hopfield
from RBM_assess import plot_confusion_matrix, rbm_features_MNIST, get_X_y_dataset
from settings import MNIST_BINARIZATION_CUTOFF, DIR_OUTPUT, CLASSIFIER, BETA


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
BATCH_SIZE = 64  # default 64
VISIBLE_UNITS = 784  # 28 x 28 images
HIDDEN_UNITS = 10  # was 128 but try 10
CD_K = 100
LEARNING_RATE = 1e-5  # default 1e-3
EPOCHS = 10  # was 10
AIS_STEPS = 10
DATA_FOLDER = 'data'
GAUSSIAN_RBM = True
LOAD_INIT_WEIGHTS = True
USE_FIELDS = False

if RBM_gaussian_custom:
    RBM = RBM_gaussian_custom
else:
    RBM = RBM_custom

########## LOADING DATASET ##########
print('Loading dataset...')
train_dataset = torchvision.datasets.MNIST(root=DATA_FOLDER, train=True, transform=torchvision.transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_dataset = torchvision.datasets.MNIST(root=DATA_FOLDER, train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

TRAINING, _ = data_mnist(binarize=True)
X, _ = get_X_y_dataset(TRAINING, dim_visible=VISIBLE_UNITS, binarize=True)

########## RBM INIT ##########
rbm = RBM(VISIBLE_UNITS, HIDDEN_UNITS, CD_K, load_init_weights=LOAD_INIT_WEIGHTS, use_fields=USE_FIELDS, learning_rate=LEARNING_RATE)
rbm.plot_model(title='epoch_0')

obj_reconstruction = np.zeros(EPOCHS)
obj_logP_termA = np.zeros(EPOCHS + 1)
obj_logP_termB = np.zeros(EPOCHS + 1)

obj_logP_termA[0] = get_obj_term_A(X, rbm.weights, rbm.visible_bias, rbm.hidden_bias, beta=BETA)
print('Estimating log Z...',)
obj_logP_termB[0] = esimate_logZ_with_AIS(rbm.weights, rbm.visible_bias, rbm.hidden_bias, beta=BETA, num_steps=AIS_STEPS)
print('INIT obj - A:', obj_logP_termA[0], '| Log Z:', obj_logP_termB[0], '| Score:', obj_logP_termA[0] - obj_logP_termB[0])

########## TRAINING RBM ##########
print('Training RBM...')
for epoch in range(EPOCHS):
    epoch_recon_error = 0.0
    for batch, _ in train_loader:
        batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data
        batch = (batch > MNIST_BINARIZATION_CUTOFF).float()    # convert to 0,1 form
        batch = -1 + batch * 2                                 # convert to -1,1 form
        batch_recon_error = rbm.contrastive_divergence(batch)
        epoch_recon_error += batch_recon_error
    rbm.plot_model(title='epoch_%d' % (epoch+1))
    print('Epoch (Reconstruction) Error (epoch=%d): %.4f' % (epoch+1, epoch_recon_error))
    obj_reconstruction[epoch] = epoch_recon_error
    obj_logP_termA[epoch+1] = get_obj_term_A(X, rbm.weights, rbm.visible_bias, rbm.hidden_bias, beta=BETA)
    print('Estimating log Z...',)
    obj_logP_termB[epoch+1] = esimate_logZ_with_AIS(rbm.weights, rbm.visible_bias, rbm.hidden_bias, beta=BETA, num_steps=AIS_STEPS)
    print('Term A:', obj_logP_termA[epoch+1], '| Log Z:', obj_logP_termB[epoch+1], '| Score:', obj_logP_termA[epoch+1] - obj_logP_termB[epoch+1])

########## PLOT AND SAVE TRAINING INFO ##########
score_arr = obj_logP_termA - obj_logP_termB

out_dir = DIR_OUTPUT + os.sep + 'logZ' + os.sep + 'rbm'
title_mod = '%dhidden_%dfields_%dcdk_%dstepsAIS_%.2fbeta' % (HIDDEN_UNITS, USE_FIELDS, CD_K, AIS_STEPS, BETA)
fpath = out_dir + os.sep + 'objective_%s' % title_mod
np.savez(fpath,
         epochs=range(EPOCHS+1),
         termA=obj_logP_termA,
         logZ=obj_logP_termB,
         score=score_arr)

plt.plot(range(EPOCHS), obj_reconstruction)
plt.xlabel('epoch'); plt.ylabel('reconstruction error')
plt.savefig(out_dir + os.sep + 'rbm_recon_%s.pdf' % (title_mod)); plt.close()

plt.plot(range(EPOCHS+1), obj_logP_termA)
plt.xlabel('epoch'); plt.ylabel(r'$- \langle H(s) \rangle$')
plt.savefig(out_dir + os.sep + 'rbm_termA_%s.pdf' % (title_mod)); plt.close()

plt.plot(range(EPOCHS+1), obj_logP_termB)
plt.xlabel('epoch'); plt.ylabel(r'$\ln \ Z$')
plt.savefig(out_dir + os.sep + 'rbm_logZ_%s.pdf' % (title_mod)); plt.close()

plt.plot(range(EPOCHS+1), score_arr)
plt.xlabel('epoch'); plt.ylabel(r'$\langle\ln \ p(x)\rangle$')
plt.savefig(out_dir + os.sep + 'rbm_score_%s.pdf' % (title_mod)); plt.close()

########## EXTRACT FEATURES ##########
print('Extracting features...')
# TODO: check classification error after each epoch
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
print('Training Classifier...')
CLASSIFIER.fit(train_features, train_labels)
print('Classifying...')
predictions = CLASSIFIER.predict(test_features).astype(int)

########## CONFUSION MATRIX ##########
confusion_matrix = np.zeros((10, 10), dtype=int)
matches = [False for _ in test_dataset]
for idx, pair in enumerate(test_dataset):
    if pair[1] == predictions[idx]:
        matches[idx] = True
    confusion_matrix[pair[1], predictions[idx]] += 1
title = "Successful test cases: %d/%d (%.3f)" % (matches.count(True), len(matches), float(matches.count(True) / len(matches)))
fpath = DIR_OUTPUT + os.sep + 'training' + os.sep + 'cm.jpg'
cm = plot_confusion_matrix(confusion_matrix, title=title, save=fpath)
print(title)
