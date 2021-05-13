from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from tqdm import tqdm
from torch import nn
import numpy as np
import torch

from evaluate import features_labels
from models.slp import SLP
from utils.utils import transform_features


def slp_train(x, y, x_test, y_test, net, optimizer, loss, epoch=100, batch_size=100):
    lv, av = [], []
    for epch in range(epoch):
        order = np.random.permutation(len(x))
        for start_index in range(0, len(x), batch_size):
            optimizer.zero_grad()
            batch_indexes = order[start_index:start_index + batch_size]
            x_batch = x[batch_indexes]
            y_batch = y[batch_indexes]
            preds = net.forward(x_batch)
            loss_value = loss(preds, y_batch)
            loss_value.backward()
            optimizer.step()
        test_preds = net.forward(x_test)
        sm = nn.Softmax(dim=1)
        a = np.argmax(sm(test_preds).detach().numpy(), axis=1)
        av += [f1_score(y_test.numpy(), a, average='weighted')]
        lv += [loss(test_preds, y_test).item()]
    return lv, av


def cross_val(x, y, clf, grid, cv=5, n_jobs=8):
    search = GridSearchCV(clf, grid, cv=cv, n_jobs=n_jobs)
    search.fit(x, y)
    return {'params': search.cv_results_['params'], 'score': search.cv_results_['mean_test_score']}


def build_classifiers_and_grids():
    classifiers = dict()
    grids = dict()
    classifiers['decision_tree'] = DecisionTreeClassifier(splitter='best', criterion='entropy')
    grids['decision_tree'] = {
        'max_depth': list(range(3, 21))
    }
    classifiers['knn'] = KNeighborsClassifier(n_jobs=8, algorithm='auto', weights='distance')
    grids['knn'] = {
        'n_neighbors': list(range(1, 15)),
        'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
    }
    classifiers['random_forest'] = RandomForestClassifier(n_jobs=8, criterion='entropy')
    grids['random_forest'] = {
        'max_depth': list(range(3, 21)),
        'n_estimators': list(range(5, 10)) + list(range(10, 100, 10)),
    }
    classifiers['svm'] = SVC(gamma='auto')
    grids['svm'] = {
        'C': np.arange(1, 21, 5),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }
    return classifiers, grids



if __name__ == '__main__':
    x, y = features_labels(f'data\\global\\train\\R.pth', scale=True)
    x_test, y_test = features_labels(f'data\\global\\test\\R.pth', scale=True)

    x = transform_features(x, 'long')
    x_test = transform_features(x_test, 'long')
    x = torch.tensor(x)
    y = torch.tensor(y)
    x_test = torch.tensor(x_test)
    y_test = torch.tensor(y_test)
    x = x.float()
    x_test = x_test.float()


    '''DT, KNN, RF, SVM cv'''
    cv_result = {}
    classifiers, grids = build_classifiers_and_grids()
    for clf in tqdm(classifiers, desc=f'DT, KNN, RF, SVM test'):
        cv_result[clf] = cross_val(x, y, classifiers[clf], grids[clf])
    torch.save(cv_result, 'data\\clfs_cv_result.pth')
    print(f'Save data\\clfs_cv_result.pth')


    '''SLP review'''
    all_loss, all_fscore = [], []
    n_in = x.shape[1]
    n_hidden = [1, 3, 5] + list(range(10, 51, 10))
    n_out = len(set(y))
    for n in tqdm(n_hidden, desc='SLP test'):
        net = SLP(n_in, n, n_out)
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=1.e-3)
        loss, acc = slp_train(x, y, x_test, y_test, net, optimizer, loss, epoch=70, batch_size=100)
        all_loss.append(loss)
        all_fscore.append(acc)
    torch.save({'loss': all_loss, 'acc': all_fscore, 'n_hidden': n_hidden}, 'data\\loss_acc_slp.pth')
    print(f'Save data\\loss_acc_slp.pth')
