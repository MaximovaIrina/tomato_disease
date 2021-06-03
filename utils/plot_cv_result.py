import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_vector(cv, name_clf, name_param, xlabel, ylabel, xticks=None):
    plt.figure(figsize=(4, 3))
    data = cv[name_clf]
    x = [param[name_param] for param in data['params']]
    score = data['score']
    plt.plot(x, score)
    plt.vlines(x=x[np.argmax(score)], ymin=plt.ylim()[0], ymax=plt.ylim()[1], linestyles='--', color='gray')
    if xticks is not None:
        plt.xticks(xticks, xticks)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def plot_matrix(cv, name_clf, name_params, xlabel, ylabel, figsize=None):
    if figsize is not None:
        plt.figure(figsize=figsize)
    data = cv[name_clf]
    x, y = [], []
    for param in data['params']:
        x += [param[name_params[0]]]
        y += [param[name_params[1]]]
    x = set(x)
    y = set(y)
    score = data['score'].reshape((len(x), len(y)))
    cmap = plt.get_cmap().__copy__()
    cmap.set_bad((1, 0, 0, 1), alpha=0.5)
    max_x, max_y = np.argmax(score) // len(y), np.argmax(score) % len(y)
    score[max_x, max_y] = np.nan
    plt.xticks(np.arange(len(y)), sorted(y))
    plt.yticks(np.arange(len(x)), sorted(x))
    plt.imshow(score, cmap=cmap, interpolation='nearest')
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def plot_slp_loss_acc(data):
    for loss, n in zip(data['loss'], data['n_hidden']):
        plt.plot(np.arange(len(loss)), loss, label=n)
    plt.legend()
    plt.xlabel('Эпоха')
    plt.ylabel('Перекрёстная энтропия')
    plt.show()

    for acc, n in zip(data['acc'], data['n_hidden']):
        plt.plot(np.arange(len(acc)), acc, label=n)
    plt.legend()
    plt.xlabel('Эпоха')
    plt.ylabel('F-score')
    plt.show()


if __name__ == '__main__':
    cv = torch.load('..\\data\\clfs_cv_result.pth')
    slp_data = torch.load('..\\data\\loss_acc_slp.pth')

    plot_vector(cv, 'decision_tree', 'max_depth', 'Глубина дерева', 'F-score', xticks=np.arange(3, 19, 3))
    plot_matrix(cv, 'knn', ['n_neighbors', 'metric'], 'Метрика', 'Число соседей', figsize=(8, 2))
    plot_matrix(cv, 'random_forest', ['max_depth', 'n_estimators'], xlabel='Число деревьев', ylabel='Глубина дерева')
    plot_matrix(cv, 'svm', ['C', 'kernel'], 'Ядро', 'Параметр регуляризации', figsize=(4, 3))
    plot_slp_loss_acc(slp_data)
