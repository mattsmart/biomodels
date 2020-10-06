import numpy as np

from data_process import image_data_collapse, binarize_image_data
from RBM_train import TRAINING, TESTING, build_models_poe
from RBM_assess import confusion_matrix_from_pred, plot_confusion_matrix
from settings import MNIST_BINARIZATION_CUTOFF, CLASSIFIER


def score_1digit_model(model, img):
    """
    'Compute numerator of Eq. 2.1...' -- Training Products of Experts
    Returns array of size dataset x 1 -- log probability (un-normed) score for each data point
    CLASSIFICATION SCORES W/o EXPAND MODELS:
        note indep of beta
        K        (1,    2,    5,    10,    20.   50,   200)
        LogReg:  (81.0, 86.2, ??.?, 92.8,  94,   94.5, 92.2)   <---- this is better than first approach (much more hidden units)
        SVD:     (91.1, ??.?, ??.?, 94.6,  95.3, 96.1, 95.3)   <---- this is similar to first approach
        LogRefre:(81.4, ??.?, ??.?, ??.?,  ??.?, ??.?, ??.?)   <---- this is similar to first approach  <---------------------------------------------------------- suddenly performing worse
        SVDre:   (75.8r,??.?, ??.?, 79.1r, ??.?, 94.8, ??.?)   <---- this is similar to first approach  <---------------------------------------------------------- suddenly performing worse
        SVDss:   (88,   ??.?, ??.?, ??.?,  94.8, 94.8, ??.?)   <---- this is similar to first approach  <---------------------------------------------------------- suddenly performing worse
        note the classifier features are always dimension 10 independent of subpattern amount
    WITH EXPAND MODELS
    note indep of beta
        K        (1,    2,    10,   20.   50,   200)
        LogReg:  (N/A,  86.6, 90.4, ??.?, ??.?, ??.?)   <---- 2 is bit better, 10 is fairly worse
        SVD:     (N/A,  ??.?, 88.6, ??.?, ??.?, ??.?)   <---- 10 is training VERY slow ~ hour; made approx got 88.6
        LogRefre:(81.4, ??.?, ??.?, ??.?, ??.?, ??.?)
        SVDre:   (75.8, ??.?, ??.?, ??.?, ??.?, ??.?)   <----- SVD is broken
        SVDss:   (??.?, ??.?, ??.?, ??.?, , ??.?)   <----- SVD is broken
        note the classifier features are size 10*K

    maybe try averaging the expanded models?
    """
    # should just be 0.5 * beta * Jij si sj
    beta = 1.0
    W = model.internal_weights
    dotp = np.dot(W.T, img)
    #score = np.dot(dotp, dotp)  #
    score = beta * np.dot(dotp, dotp) / 2.0
    return score


def classifier_on_rbm_scores(models, dataset_train, dataset_test, clfs=None):
    """
    TODO: store progress on heirarchical clustering; that is the slowest step
    clfs: list of classifiers
    """
    if clfs is None:
        clfs = [CLASSIFIER]

    features_order = list(models.keys())
    feature_dim = len(features_order)

    def get_X_y_features(dataset):
        X = np.zeros((len(dataset), feature_dim))
        y = np.zeros(len(dataset), dtype=int)
        for idx, pair in enumerate(dataset):
            elem_arr, elem_label = pair
            preprocessed_input = binarize_image_data(image_data_collapse(elem_arr), threshold=MNIST_BINARIZATION_CUTOFF)
            features = np.array([score_1digit_model(models[key], preprocessed_input) for key in features_order])
            #features = np.array([score_1digit_model(models[idx], preprocessed_input) for idx in range(10)])

            X[idx, :] = features
            y[idx] = elem_label
        return X, y

    print("[classifier_on_rbm_features] Step 1: get features for training")
    X_train_reduced, y_train = get_X_y_features(dataset_train)
    print("\tTraining data dimension", X_train_reduced.shape, y_train.shape)

    print("[classifier_on_rbm_features] Step 2: train classifier layer")
    for clf in clfs:
        print('fitting...')
        clf.fit(X_train_reduced, y_train)  # fit data

    print("[classifier_on_rbm_features] Step 3: get features for testing")
    X_test_reduced, y_test = get_X_y_features(dataset_test)

    print("[classifier_on_rbm_features] Step 4: classification metrics and confusion matrix")
    cms = [0] * len(clfs)
    accs = [0] * len(clfs)
    for idx, clf in enumerate(clfs):
        print('predicting...')
        predictions = clf.predict(X_test_reduced).astype(int)
        confusion_matrix, matches = confusion_matrix_from_pred(predictions, y_test)
        acc = float(matches.count(True) / len(matches))
        cms[idx] = confusion_matrix
        accs[idx] = acc
        print("Successful test cases: %d/%d (%.3f)" % (matches.count(True), len(matches), acc))
    return cms, accs


if __name__ == '__main__':
    #TRAINING = TRAINING[0:10000]  # take subset for faster evaluation  TODO care using subset
    USE_SVM = False
    
    ks = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
    accs = [0] * len(ks)

    if USE_SVM:
        from sklearn import svm
        CLA = svm.SVC(gamma=0.9*1e-4, C=6)
        CLB = svm.SVC(gamma=0.93*1e-4, C=6)
        CLC = svm.SVC(gamma=0.97*1e-4, C=6)
        CLD = svm.SVC(gamma=1.04*1e-4, C=6)
        CLE = svm.SVC(gamma=1.08*1e-4, C=6)
        CLF = svm.SVC(gamma=1.11*1e-4, C=6)
    else:
        from sklearn.linear_model import LogisticRegression
        CLA = LogisticRegression(C=1e5, multi_class='multinomial', penalty='l1', solver='saga', tol=0.001)
        CLB = LogisticRegression(C=1e3, multi_class='multinomial', penalty='l1', solver='saga', tol=0.001)
        CLC = LogisticRegression(C=1e1, multi_class='multinomial', penalty='l1', solver='saga', tol=0.001)
        CLD = LogisticRegression(C=1e5, multi_class='multinomial', penalty='l2', solver='saga', tol=0.001)
        CLE = LogisticRegression(C=1e3, multi_class='multinomial', penalty='l2', solver='saga', tol=0.001)
        CLF = LogisticRegression(C=1e1, multi_class='multinomial', penalty='l2', solver='saga', tol=0.001)    
    clfs = [CLA]
    #clfs = [CLA, CLB, CLC, CLD, CLE, CLF]
    assert len(clfs) == 1  # refactor code so we can have a list of clfs and a list of subpattern ints

    for idx, k in enumerate(ks):
        print("SCORING k_pattern", k)
        models = build_models_poe(TRAINING, k_pattern=k)
        list_confusion_matrix, list_acc = classifier_on_rbm_scores(models, TRAINING, TESTING, clfs=clfs)
        for cm in list_confusion_matrix:
            plot_confusion_matrix(cm)  # only gets CM for the last clf
        accs[idx] = list_acc[0]
    print(ks)
    print(accs)
