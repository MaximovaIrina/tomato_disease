from sklearn.metrics import precision_recall_fscore_support
import os

from models.dtree import DTree
from utils import features_labels, transform_features

if __name__ == '__main__':
    file = os.path.join(os.getcwd(), '..\\data\\global\\train\\R.pth')
    test_file = os.path.join(os.getcwd(), '..\\data\\global\\test\\R.pth')

    x, y = features_labels(file, scale=True)
    x_test, y_test = features_labels(test_file, scale=True)

    a, b = 8, 68
    x = x[:, a:b]
    x_test = x_test[:, a:b]

    x = transform_features(x, 'long')
    x_test = transform_features(x_test, 'long')

    print('MultiTest:')
    dt = DTree(lim_std=2)
    dt.fit(x, y)
    predict = dt.predict(x_test)
    pr, rec, fs, _, = precision_recall_fscore_support(y_test, predict, average='weighted')
    print(f'precision = {int(pr * 1000) / 10}, '
          f'recall = {int(rec * 1000) / 10}, '
          f'fscore = {int(fs * 1000) / 10}')

    print('\nBinaryTest:')
    y = [label == 2 for label in y]
    y_test = [label == 2 for label in y_test]
    dt = DTree(lim_std=2.5)
    dt.fit(x, y)
    predict = dt.predict(x_test)
    pr, rec, fs, _, = precision_recall_fscore_support(y_test, predict, average='binary')
    print(f'precision = {int(pr * 1000) / 10}, '
          f'recall = {int(rec * 1000) / 10}, '
          f'fscore = {int(fs * 1000) / 10}')
