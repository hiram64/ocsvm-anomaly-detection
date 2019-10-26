import argparse
from collections import namedtuple
import sys

import numpy as np
from sklearn import svm
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score


def parse_args():
    parser = argparse.ArgumentParser(description='Anomaly Detection by One Class SVM')

    parser.add_argument('--data_path', default='./data/cifar10_cae.npz', type=str, help='path to dataset')
    parser.add_argument('--normal_label', default=8, type=int,
                        help='label defined as normal. Other classes are treated as abnormal')
    parser.add_argument('--rate_normal_train', default=0.82, type=float, help='rate of normal data to use in training')
    parser.add_argument('--rate_anomaly_test', default=0.1, type=float,
                        help='rate of abnormal data versus normal data in test data. The default setting is 10:1(=0.1)')
    parser.add_argument('--test_rep_count', default=10, type=int,
                        help='counts of test repeats per one trained model. For a model, test data selection and evaluation are repeated.')
    parser.add_argument('--TRAIN_RAND_SEED', default=42, type=int, help='random seed used selecting training data')
    parser.add_argument('--TEST_RAND_SEED', default=[42, 89, 2, 156, 491, 32, 67, 341, 100, 279], type=list,
                        help='random seed used selecting test data.'
                             'The number of elements should equal to "test_rep_count" for reproductivity of validation.'
                             'When the length of this list is less than "test_rep_count", seed is randomly generated')

    args = parser.parse_args()

    return args


def load_data(data_to_path):
    """load data
    data should be compressed in npz
    """
    data = np.load(data_to_path)

    try:
        full_images = data['ae_out']
        full_labels = data['labels']
    except:
        print('Loading data should be numpy array and has "ae_out" and "labels" keys.')
        sys.exit(1)

    return full_images, full_labels


def prepare_data(full_images, full_labels, normal_label, rate_normal_train, TRIN_RAND_SEED):
    """prepare data
    split data into anomaly data and normal data
    """
    TRAIN_DATA_RNG = np.random.RandomState(TRIN_RAND_SEED)

    # data whose label corresponds to anomaly label, otherwise treated as normal data
    ano_x = full_images[full_labels != normal_label]
    ano_y = full_labels[full_labels != normal_label]
    normal_x = full_images[full_labels == normal_label]
    normal_y = full_labels[full_labels == normal_label]

    # replace label : anomaly -> 1 : normal -> 0
    ano_y[:] = 1
    normal_y[:] = 0

    # shuffle normal data and label
    inds = TRAIN_DATA_RNG.permutation(normal_x.shape[0])
    normal_x_data = normal_x[inds]
    normal_y_data = normal_y[inds]

    # split normal data into train and test
    index = int(normal_x.shape[0] * rate_normal_train)
    trainx = normal_x_data[:index]
    testx_n = normal_x_data[index:]
    testy_n = normal_y_data[index:]

    split_data = namedtuple('split_data', ('train_x', 'testx_n', 'testy_n', 'ano_x', 'ano_y'))

    return split_data(
        train_x=trainx,
        testx_n=testx_n,
        testy_n=testy_n,
        ano_x=ano_x,
        ano_y=ano_y
    )


def make_test_data(split_data, RNG, rate_anomaly_test):
    """make test data which has specified mixed rate(rate_anomaly_test).
    shuffle and concatenate normal and abnormal data"""

    ano_x = split_data.ano_x
    ano_y = split_data.ano_y
    testx_n = split_data.testx_n
    testy_n = split_data.testy_n

    # anomaly data in test
    inds_1 = RNG.permutation(ano_x.shape[0])
    ano_x = ano_x[inds_1]
    ano_y = ano_y[inds_1]

    index_1 = int(testx_n.shape[0] * rate_anomaly_test)
    testx_a = ano_x[:index_1]
    testy_a = ano_y[:index_1]

    # concatenate test normal data and test anomaly data
    testx = np.concatenate([testx_a, testx_n], axis=0)
    testy = np.concatenate([testy_a, testy_n], axis=0)

    return testx, testy


def calc_metrics(testy, scores):
    precision, recall, _ = precision_recall_curve(testy, scores)
    roc_auc = roc_auc_score(testy, scores)
    prc_auc = auc(recall, precision)

    return roc_auc, prc_auc


def main():
    # set parameters
    args = parse_args()
    data_path = args.data_path
    normal_label = args.normal_label
    rate_normal_train = args.rate_normal_train
    rate_anomaly_test = args.rate_anomaly_test
    test_rep_count = args.test_rep_count
    TRAIN_RAND_SEED = args.TRAIN_RAND_SEED
    TEST_RAND_SEED = args.TEST_RAND_SEED

    # load and prepare data
    full_images, full_labels = load_data(data_path)
    split_data = prepare_data(full_images, full_labels, normal_label, rate_normal_train, TRAIN_RAND_SEED)

    pr_scores = []
    roc_scores = []

    # nu : the upper limit ratio of anomaly data(0<=nu<=1)
    nus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # train model and evaluate with changing parameter nu
    for nu in nus:
        # train with nu
        clf = svm.OneClassSVM(nu=nu, kernel='rbf', gamma='auto')
        clf.fit(split_data.train_x)

        total_pr = 0
        total_roc = 0

        # repeat test by randomly selected data and evaluate
        for j in range(test_rep_count):
            # select test data and test
            if j < len(TEST_RAND_SEED):
                TEST_SEED = np.random.RandomState(TEST_RAND_SEED[j])
            else:
                TEST_SEED = np.random.RandomState(np.random.randint(0, 10000))

            testx, testy = make_test_data(split_data, TEST_SEED, rate_anomaly_test)
            scores = clf.decision_function(testx).ravel() * (-1)

            # calculate evaluation metrics
            roc_auc, prc_auc = calc_metrics(testy, scores)

            total_pr += prc_auc
            total_roc += roc_auc

        # calculate average
        total_pr /= test_rep_count
        total_roc /= test_rep_count

        pr_scores.append(total_pr)
        roc_scores.append(total_roc)

        print('--- nu : ', nu, ' ---')
        print('PR AUC : ', total_pr)
        print('ROC_AUC : ', total_roc)

    print('***' * 5)
    print('PR_AUC MAX : ', max(pr_scores))
    print('ROC_AUC MAX : ', max(roc_scores))
    print('ROC_MAX_NU : ', nus[int(np.argmax(roc_scores))])


if __name__ == '__main__':
    main()
