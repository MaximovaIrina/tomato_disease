from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from tqdm import tqdm
import pandas as pd
import torch
import time

from models.dtree import DTree
from models.fisher import Fisher
from models.slp import SLPClassifer
from utils.utils import scale, features_labels, transform_features


def build_classifiers():
    classifiers = dict()
    classifiers['MDT'] = DTree(lim_std=2)
    classifiers['Fisher'] = Fisher()
    classifiers['DT'] = DecisionTreeClassifier(criterion='entropy', max_depth=9)
    classifiers['KNN'] = KNeighborsClassifier(metric='euclidean', n_jobs=8, n_neighbors=3, weights='distance')
    classifiers['RF'] = RandomForestClassifier(n_jobs=8, criterion='entropy', max_depth=15, n_estimators=80)
    classifiers['SVM'] = SVC(C=10, gamma='auto')
    classifiers['SLP'] = None
    return classifiers


def train_and_test(ind, clf, train_data, test_data):
    x_train, y_train = train_data
    x_test, y_test = test_data

    x_train = x_train[:, ind]
    x_test = x_test[:, ind]

    x_train, mean_train, std_train = scale(x_train)
    x_test = scale(x_test, mean_train, std_train)

    t0 = time.perf_counter()
    clf.fit(x_train, y_train)
    t1 = time.perf_counter()
    print(f"Time {lenn} train {channel} {clf.__class__.__name__}: ", t1-t0)

    prediction = clf.predict(x_test)

    classifier_state = {'classifier': clf, 'mean': mean_train, 'std': std_train}
    torch.save(classifier_state, f'data\\state\\{clf.__class__.__name__}_{mode}_{channel}.pth')
    torch.save({'prediction': prediction, 'y': y_test},
               f'data\\test_predict\\{clf.__class__.__name__}_{mode}_{channel}.pth')

    if len(set(y_train)) == 2:
        fsocre = f1_score(y_test, prediction, average='binary')
    else:
        fsocre = f1_score(y_test, prediction, average='weighted')
    return {'state': clf, 'mean': mean_train, 'std': std_train, 'ind': ind}, fsocre


def classification_results(classifiers, train, test):
    ind = [list(range(4)), list(range(8)), list(range(8, len(train[0][0]))), list(range(len(train[0][0])))]
    properties = ('STAT', 'STAT+HIST', 'GLCM', 'FEATURES')
    data = {'name': [], 'properties': [], 'fscore': []}

    for name, clf in tqdm(classifiers.items()):
        for i, prop in enumerate(properties):
            if name == 'SLP':
                clf = SLPClassifer(n_in=len(ind[i]), n_hidden=50, n_out=len(set(train[1])), epoch=100, batch_size=100)
            clf_state, fscore = train_and_test(ind=ind[i], clf=clf, train_data=train, test_data=test)
            data['name'] += [name]
            data['properties'] += [prop]
            data['fscore'] += [fscore]
    return data


if __name__ == '__main__':
    task = ['detection']
    modes = ['global']
    channels = ['R', 'ndvi']
    lenght = ['short']

    for t in task:
        for mode in modes:
            for channel in channels:
                for lenn in lenght:
                    x, y = features_labels(f'data\\{mode}\\train\\{channel}.pth')
                    x_test, y_test = features_labels(f'data\\{mode}\\test\\{channel}.pth')

                    if t == 'detection':
                        y = [1 if label == 2 else 0 for label in y]
                        y_test = [1 if label == 2 else 0 for label in y_test]

                    x = transform_features(x, lenn)
                    x_test = transform_features(x_test, lenn)

                    classifiers = build_classifiers()
                    data = classification_results(classifiers, train=(x, y), test=(x_test, y_test))

                    print(f"{mode}_{channel}_{lenn}", data)

                    df = pd.DataFrame(data=data)
                    if t == 'classification':
                        df.to_excel(f'data\\result\\{mode}_{channel}_{lenn}.xlsx')
                        print(f'Save data\\result\\{mode}_{channel}_{lenn}.xlsx')
                    else:
                        df.to_excel(f'data\\result_detect\\{mode}_{channel}_{lenn}.xlsx')
                        print(f'Save data\\result_detect\\{mode}_{channel}_{lenn}.xlsx')
